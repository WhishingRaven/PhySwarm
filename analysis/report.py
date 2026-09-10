"""PhySwarm CSV를 다각도의 재현 가능한 그래프와 요약으로 변환한다.

기존 ``controllers/*/plot`` 스크립트는 파일명과 논문 그림 하나에 강하게
결합되어 있고 import 즉시 실행되는 경우가 많다. 이 분석기는 입력 파일과
출력 디렉터리를 명시적으로 받고, 동일한 데이터에는 동일한 산출물을 만든다.

지원 입력
---------
* trajectory 로그: ``step, agent_id, pos_x, pos_y``를 포함하는 CSV
* training/evaluation 로그: ``step``과 수치 metric 열을 포함하는 CSV

출력은 PNG 그래프, machine-readable ``summary.json``, 그리고 trajectory의
경우 집계된 ``per_step.csv``다. 원본 데이터는 절대 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# headless CI와 Webots fast mode에서도 같은 renderer를 사용한다. Matplotlib가
# 사용자 홈에 cache를 쓰지 못하는 환경을 고려해 임시 cache 위치를 지정한다.
_MPL_CACHE = Path(tempfile.gettempdir()) / "physwarm-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.metrics import (
    collision_rate,
    control_smoothness,
    foraging_efficiency,
    navigation_arrival_mask,
    navigation_time,
    relay_chain_connectivity,
    time_to_connectivity,
    transport_economy,
)
from tasks.specs import PAPER_SCENARIOS, ScenarioSpec, get_scenario


TRAJECTORY_REQUIRED_COLUMNS = {"step", "agent_id", "pos_x", "pos_y"}
PARAMETER_PREFIXES = ("w_", "k_diff", "lambda_", "beta", "alpha")


@dataclass(frozen=True)
class AnalysisResult:
    """분석 API가 생성한 파일과 핵심 요약."""

    input_path: Path
    output_directory: Path
    kind: str
    scenario: str | None
    generated_files: tuple[Path, ...]
    summary: dict[str, object]


def _finite_number(value: object) -> float | int | None:
    """JSON에 NaN/Inf가 들어가지 않도록 안전한 Python 숫자로 변환한다."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    """사람과 후속 스크립트가 모두 읽을 수 있는 안정적인 JSON을 기록한다."""

    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _configure_plot_style() -> None:
    """논문/문서 양쪽에서 읽기 쉬운 공통 시각 스타일."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.dpi": 180,
        }
    )


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    """layout을 정리해 저장하고 figure handle을 즉시 닫는다."""

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _infer_scenario(columns: Iterable[str]) -> ScenarioSpec | None:
    """서로 겹치지 않는 로그 열로 과제를 보수적으로 판별한다."""

    names = set(columns)
    if {"is_carrying", "w_food", "w_nest"} & names:
        return PAPER_SCENARIOS["foraging"]
    if {"w_flow", "w_shape", "corridor_width", "q_shape"} & names:
        return PAPER_SCENARIOS["navigation"]
    if {"role", "chain_connected", "w_target", "w_center"} & names:
        return PAPER_SCENARIOS["rescue"]
    return None


def _resolve_scenario(
    requested: str,
    columns: Iterable[str],
) -> ScenarioSpec | None:
    if requested != "auto":
        return get_scenario(requested)
    return _infer_scenario(columns)


def _numeric_columns(frame: pd.DataFrame, *, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _aligned_agent_tensor(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """agent-row columns를 완전한 ``(T,N,C)`` tensor로 정렬한다.

    Smoothness나 centroid metric은 빠진 robot-step을 0으로 채우면 값이 왜곡된다.
    모든 요청 열에 공통인 step/agent만 고르는 대신, 원래 grid가 완전하지 않으면
    ``None``을 반환해 caller가 metric을 생략하도록 한다.
    """

    if not set(columns).issubset(frame.columns):
        return None
    steps = np.sort(frame["step"].unique())
    agents = np.sort(frame["agent_id"].unique())
    if steps.size == 0 or agents.size == 0:
        return None
    tensors = []
    for column in columns:
        table = frame.pivot_table(
            index="step",
            columns="agent_id",
            values=column,
            aggfunc="mean",
        ).reindex(index=steps, columns=agents)
        values = table.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            return None
        tensors.append(values)
    return steps.astype(np.float64), agents, np.stack(tensors, axis=-1)


def _parameter_columns(frame: pd.DataFrame) -> list[str]:
    """trajectory와 training 로그의 두 naming convention을 모두 찾는다."""

    columns: list[str] = []
    for column in frame.columns:
        leaf = column.split("/")[-1]
        if column.startswith("Param_") or leaf.startswith(PARAMETER_PREFIXES):
            if pd.api.types.is_numeric_dtype(frame[column]):
                columns.append(column)
    return columns


def _outcome_columns(frame: pd.DataFrame) -> list[str]:
    candidates = (
        "reward",
        "score",
        "collision",
        "delivery",
        "pickup",
        "rescued",
        "chain",
        "error",
        "efficiency",
        "throughput",
        "loss",
    )
    return [
        column
        for column in _numeric_columns(frame, exclude={"step", "agent_id"})
        if any(token in column.lower() for token in candidates)
    ]


def _pairwise_distance_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """각 step의 살아 있는 로봇 간 거리 중앙값/최솟값을 계산한다."""

    records: list[dict[str, float]] = []
    for step, group in frame.groupby("step", sort=True):
        if "is_alive" in group:
            group = group[group["is_alive"] > 0]
        points = group[["pos_x", "pos_y"]].to_numpy(dtype=float)
        if len(points) < 2:
            records.append(
                {"step": float(step), "pair_distance_min": np.nan, "pair_distance_median": np.nan}
            )
            continue
        delta = points[:, None, :] - points[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        upper_triangle = distance[np.triu_indices(len(points), k=1)]
        records.append(
            {
                "step": float(step),
                "pair_distance_min": float(np.min(upper_triangle)),
                "pair_distance_median": float(np.median(upper_triangle)),
            }
        )
    return pd.DataFrame.from_records(records)


def _aggregate_trajectory(frame: pd.DataFrame, scenario: ScenarioSpec | None) -> pd.DataFrame:
    """agent-row 로그를 그래프에 적합한 step-row 시계열로 집계한다."""

    grouped = frame.groupby("step", sort=True)
    per_step = pd.DataFrame(index=sorted(frame["step"].unique()))
    per_step.index.name = "step"

    if "is_alive" in frame:
        per_step["alive_fraction"] = grouped["is_alive"].mean()
    else:
        per_step["alive_fraction"] = 1.0
    if "is_colliding" in frame:
        per_step["collision_fraction"] = grouped["is_colliding"].mean()
    if "pickup_event" in frame:
        per_step["pickups"] = grouped["pickup_event"].sum()
        per_step["pickups_cumulative"] = per_step["pickups"].cumsum()
    if "delivery_event" in frame:
        per_step["deliveries"] = grouped["delivery_event"].sum()
        per_step["deliveries_cumulative"] = per_step["deliveries"].cumsum()
    if "rescued_count" in frame:
        per_step["rescued_count"] = grouped["rescued_count"].max()
    if "chain_connected" in frame:
        per_step["chain_connected"] = grouped["chain_connected"].max()

    for column in _parameter_columns(frame):
        per_step[f"mean:{column}"] = grouped[column].mean()
        per_step[f"q25:{column}"] = grouped[column].quantile(0.25)
        per_step[f"q75:{column}"] = grouped[column].quantile(0.75)

    # 이산 phase를 비율로 바꿔 swarm-level reaction 변화를 볼 수 있게 한다.
    if "role" in frame:
        phase_names = scenario.phases if scenario is not None else ("0", "1", "2")
        for role_index, phase_name in enumerate(phase_names):
            per_step[f"phase:{phase_name}"] = grouped["role"].apply(
                lambda values, index=role_index: float(np.mean(values == index))
            )
    elif "is_carrying" in frame:
        per_step["phase:carrying"] = grouped["is_carrying"].mean()

    pairwise = _pairwise_distance_statistics(frame).set_index("step")
    per_step = per_step.join(pairwise, how="left")
    return per_step.reset_index()


def _trajectory_summary(
    frame: pd.DataFrame,
    per_step: pd.DataFrame,
    scenario: ScenarioSpec | None,
) -> dict[str, object]:
    """논문 지표와 데이터 품질 정보를 한 JSON 객체로 정리한다."""

    step_min = float(frame["step"].min())
    step_max = float(frame["step"].max())
    episode_duration = max(step_max - step_min + 1.0, 1.0)
    num_agents = int(frame["agent_id"].nunique())
    summary: dict[str, object] = {
        "kind": "trajectory",
        "scenario": scenario.key if scenario else None,
        "rows": int(len(frame)),
        "num_agents": num_agents,
        "step_min": _finite_number(step_min),
        "step_max": _finite_number(step_max),
        "episode_duration_steps": _finite_number(episode_duration),
        "duplicate_agent_steps": int(frame.duplicated(["step", "agent_id"]).sum()),
        "missing_numeric_values": int(frame.select_dtypes(include=[np.number]).isna().sum().sum()),
    }
    if "is_alive" in frame:
        summary["final_alive_fraction"] = _finite_number(
            per_step["alive_fraction"].iloc[-1]
        )
    if "is_colliding" in frame:
        summary["collision_rate"] = collision_rate(frame["is_colliding"].to_numpy())

    command_candidates = (
        ("cmd_omega_l", "cmd_omega_r"),
        ("left_speed", "right_speed"),
    )
    if summary["duplicate_agent_steps"] == 0:
        for command_columns in command_candidates:
            aligned_commands = _aligned_agent_tensor(frame, command_columns)
            if aligned_commands is None:
                continue
            command_steps, command_agents, commands = aligned_commands
            active = None
            aligned_active = _aligned_agent_tensor(frame, ("is_alive",))
            if aligned_active is not None:
                active_steps, active_agents, active_values = aligned_active
                if (
                    np.array_equal(command_steps, active_steps)
                    and np.array_equal(command_agents, active_agents)
                ):
                    active = active_values[..., 0] > 0.0
            summary["control_smoothness"] = control_smoothness(
                commands, active_mask=active
            )
            summary["control_smoothness_columns"] = list(command_columns)
            break

    if scenario and scenario.key == "foraging":
        deliveries = (
            float(frame["delivery_event"].sum()) if "delivery_event" in frame else 0.0
        )
        summary["delivered_resources"] = _finite_number(deliveries)
        summary["foraging_efficiency"] = foraging_efficiency(
            deliveries, num_agents, episode_duration
        )

        # 각 agent의 시간축을 정렬해 식 S178에 필요한 (T,N,2)를 만든다.
        x_table = frame.pivot_table(index="step", columns="agent_id", values="pos_x")
        y_table = frame.pivot_table(index="step", columns="agent_id", values="pos_y")
        common_steps = x_table.index.intersection(y_table.index)
        common_agents = x_table.columns.intersection(y_table.columns)
        tracks = np.stack(
            (
                x_table.loc[common_steps, common_agents].to_numpy(),
                y_table.loc[common_steps, common_agents].to_numpy(),
            ),
            axis=-1,
        )
        target_pairs = [
            (column, column.replace("_x", "_y"))
            for column in frame.columns
            if column.startswith("target_") and column.endswith("_x")
            and column.replace("_x", "_y") in frame
        ]
        if {"nest_x", "nest_y"}.issubset(frame.columns) and target_pairs:
            first = frame.iloc[0]
            nest = np.array([first["nest_x"], first["nest_y"]], dtype=float)
            distances = [
                np.linalg.norm(
                    np.array([first[x_column], first[y_column]], dtype=float) - nest
                )
                for x_column, y_column in target_pairs
            ]
            summary["transport_economy"] = transport_economy(
                tracks, deliveries, float(np.mean(distances))
            )

    if scenario and scenario.key == "navigation":
        for column in ("q_shape", "q_total", "score", "corridor_width"):
            if column in frame:
                summary[f"mean_{column}"] = _finite_number(frame[column].mean())
        if "pair_distance_min" in per_step:
            summary["minimum_pair_distance"] = _finite_number(
                per_step["pair_distance_min"].min()
            )
        aligned_positions = _aligned_agent_tensor(frame, ("pos_x", "pos_y"))
        navigation_columns = {"goal_x", "goal_y", "goal_threshold"}
        if aligned_positions is not None and navigation_columns.issubset(frame.columns):
            position_steps, position_agents, positions = aligned_positions
            first = frame.iloc[0]
            active = None
            aligned_active = _aligned_agent_tensor(frame, ("is_alive",))
            if aligned_active is not None:
                active_steps, active_agents, active_values = aligned_active
                if (
                    np.array_equal(position_steps, active_steps)
                    and np.array_equal(position_agents, active_agents)
                ):
                    active = active_values[..., 0] > 0.0
            max_time = (
                float(first["episode_max_steps"])
                if "episode_max_steps" in frame
                else float(position_steps[-1])
            )
            goal = np.array([first["goal_x"], first["goal_y"]], dtype=float)
            threshold = float(first["goal_threshold"])
            arrivals = navigation_arrival_mask(
                positions,
                goal,
                threshold,
                active_mask=active,
            )
            summary["navigation_time"] = navigation_time(
                positions,
                goal,
                threshold,
                steps=position_steps,
                max_time=max_time,
                active_mask=active,
            )
            summary["navigation_time_censored"] = not bool(arrivals.any())

    if scenario and scenario.key == "rescue":
        if "rescued_count" in frame:
            summary["rescued_targets"] = _finite_number(frame["rescued_count"].max())
        if "chain_connected" in frame:
            connected_steps = per_step.loc[per_step["chain_connected"] > 0, "step"]
            summary["task_success"] = bool(len(connected_steps))
            max_time = (
                float(frame["episode_max_steps"].iloc[0])
                if "episode_max_steps" in frame
                else step_max
            )
            summary["time_to_connectivity"] = time_to_connectivity(
                per_step["chain_connected"].to_numpy(dtype=float),
                steps=per_step["step"].to_numpy(dtype=float),
                max_time=max_time,
            )
            summary["time_to_connectivity_censored"] = not bool(len(connected_steps))
        lambda2_values: list[float] = []
        required = {
            "base_x",
            "base_y",
            "target_0_x",
            "target_0_y",
            "comm_view",
        }
        if required.issubset(frame.columns):
            for _, group in frame.groupby("step", sort=True):
                first = group.iloc[0]
                active = np.ones(len(group), dtype=bool)
                if "is_alive" in group:
                    active &= group["is_alive"].to_numpy() > 0
                if "role" in group:
                    # 현재 구조 과제는 target 정보를 받은 responder/relay만 체인 노드다.
                    active &= group["role"].to_numpy() > 0
                lambda2_values.append(
                    relay_chain_connectivity(
                        group[["pos_x", "pos_y"]].to_numpy(),
                        np.array([first["base_x"], first["base_y"]]),
                        np.array([first["target_0_x"], first["target_0_y"]]),
                        float(first["comm_view"]),
                        active_mask=active,
                    )
                )
            summary["relay_lambda2_max"] = _finite_number(max(lambda2_values, default=0.0))
            summary["relay_lambda2_mean"] = _finite_number(
                np.mean(lambda2_values) if lambda2_values else 0.0
            )
    return summary


def _plot_trajectories(
    frame: pd.DataFrame,
    output_directory: Path,
    scenario: ScenarioSpec | None,
) -> Path:
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    color_map = plt.get_cmap("tab20")
    for color_index, (agent_id, track) in enumerate(frame.groupby("agent_id")):
        track = track.sort_values("step")
        if "is_alive" in track:
            track = track[track["is_alive"] > 0]
        if track.empty:
            continue
        color = color_map(color_index % 20)
        axis.plot(track["pos_x"], track["pos_y"], color=color, alpha=0.72, lw=1.1)
        axis.scatter(track["pos_x"].iloc[0], track["pos_y"].iloc[0], s=18, color=color, marker="o")
        axis.scatter(track["pos_x"].iloc[-1], track["pos_y"].iloc[-1], s=22, color=color, marker="x")

    first = frame.iloc[0]
    anchors = [
        ("nest", "nest_x", "nest_y", "s", "#2e7d32"),
        ("base", "base_x", "base_y", "s", "#1565c0"),
    ]
    for label, x_column, y_column, marker, color in anchors:
        if x_column in frame and y_column in frame:
            axis.scatter(first[x_column], first[y_column], s=90, marker=marker, color=color, label=label)
    for x_column in [column for column in frame if column.startswith("target_") and column.endswith("_x")]:
        y_column = x_column.replace("_x", "_y")
        if y_column in frame:
            axis.scatter(first[x_column], first[y_column], s=90, marker="*", color="#c62828", label=x_column[:-2])

    title = scenario.title if scenario else "PhySwarm"
    axis.set_title(f"{title} - robot trajectories")
    axis.set_xlabel("x position [m]")
    axis.set_ylabel("y position [m]")
    axis.set_aspect("equal", adjustable="datalim")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        # 동적 target가 여러 개여도 같은 label은 한 번만 표시한다.
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="best")
    return _save_figure(figure, output_directory / "trajectories.png")


def _plot_swarm_dynamics(
    per_step: pd.DataFrame,
    output_directory: Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    step = per_step["step"]

    axes[0, 0].plot(step, per_step["alive_fraction"], label="alive fraction")
    if "collision_fraction" in per_step:
        axes[0, 0].plot(step, per_step["collision_fraction"], label="collision fraction")
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 0].set_title("Safety and resilience")
    axes[0, 0].legend()

    progress_columns = [
        column
        for column in (
            "pickups_cumulative",
            "deliveries_cumulative",
            "rescued_count",
            "chain_connected",
        )
        if column in per_step
    ]
    for column in progress_columns:
        axes[0, 1].plot(step, per_step[column], label=column.replace("_", " "))
    axes[0, 1].set_title("Task progress")
    if progress_columns:
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, "No event columns", ha="center", va="center", transform=axes[0, 1].transAxes)

    phase_columns = [column for column in per_step if column.startswith("phase:")]
    for column in phase_columns:
        axes[1, 0].plot(step, per_step[column], label=column.split(":", 1)[1])
    axes[1, 0].set_title("Behavioral phase composition")
    axes[1, 0].set_ylim(-0.03, 1.03)
    if phase_columns:
        axes[1, 0].legend()

    axes[1, 1].plot(step, per_step["pair_distance_median"], label="median")
    axes[1, 1].plot(step, per_step["pair_distance_min"], label="minimum", alpha=0.8)
    axes[1, 1].set_title("Inter-robot spacing")
    axes[1, 1].set_ylabel("distance [m]")
    axes[1, 1].legend()
    for axis in axes[1]:
        axis.set_xlabel("simulation step")
    return _save_figure(figure, output_directory / "swarm_dynamics.png")


def _plot_parameter_evolution(
    per_step: pd.DataFrame,
    output_directory: Path,
) -> Path | None:
    mean_columns = [column for column in per_step if column.startswith("mean:")]
    if not mean_columns:
        return None
    grouped_columns: dict[str, list[str]] = {
        "Advection weights": [],
        "Diffusion and reaction": [],
        "Task-specific geometry": [],
    }
    for mean_column in mean_columns:
        original_name = mean_column.split(":", 1)[1]
        leaf_name = original_name.split("/")[-1]
        if leaf_name.startswith("w_"):
            grouped_columns["Advection weights"].append(mean_column)
        elif leaf_name.startswith(("k_diff", "lambda_")):
            grouped_columns["Diffusion and reaction"].append(mean_column)
        else:
            grouped_columns["Task-specific geometry"].append(mean_column)
    grouped_columns = {
        title: columns for title, columns in grouped_columns.items() if columns
    }

    # beta처럼 범위가 1--5인 값과 0.0x 크기의 장 가중치를 같은 축에 두면
    # 작은 파라미터 변화가 보이지 않는다. 물리 단위별 subplot으로 분리한다.
    figure, axes = plt.subplots(
        len(grouped_columns),
        1,
        figsize=(11.0, 2.8 * len(grouped_columns)),
        sharex=True,
    )
    axes_array = np.atleast_1d(axes)
    step = per_step["step"].to_numpy()
    for axis, (group_title, columns) in zip(axes_array, grouped_columns.items()):
        for mean_column in columns:
            original_name = mean_column.split(":", 1)[1]
            values = per_step[mean_column].to_numpy(dtype=float)
            axis.plot(step, values, label=original_name, lw=1.5)
            q25 = f"q25:{original_name}"
            q75 = f"q75:{original_name}"
            if q25 in per_step and q75 in per_step:
                axis.fill_between(
                    step,
                    per_step[q25].to_numpy(dtype=float),
                    per_step[q75].to_numpy(dtype=float),
                    alpha=0.10,
                )
        axis.set_title(group_title)
        axis.set_ylabel("projected value")
        axis.legend(ncol=min(4, max(1, len(columns))), loc="best")
    axes_array[-1].set_xlabel("simulation step")
    figure.suptitle("Physical parameter evolution (mean and interquartile range)")
    return _save_figure(figure, output_directory / "physical_parameters.png")


def _plot_correlations(frame: pd.DataFrame, output_directory: Path) -> Path | None:
    selected = list(dict.fromkeys(_parameter_columns(frame) + _outcome_columns(frame)))
    # 지나치게 큰 heatmap은 읽기 어려우므로 분산이 있는 첫 18개 지표만 사용한다.
    selected = [column for column in selected if frame[column].nunique(dropna=True) > 1][:18]
    if len(selected) < 2:
        return None
    correlation = frame[selected].corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(max(7.0, len(selected) * 0.55), max(6.0, len(selected) * 0.5)))
    image = axis.imshow(correlation.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(len(selected)), labels=selected, rotation=60, ha="right")
    axis.set_yticks(range(len(selected)), labels=selected)
    axis.set_title("Parameter/outcome Pearson correlations")
    figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03, label="correlation")
    return _save_figure(figure, output_directory / "parameter_correlations.png")


def _analyze_trajectory(
    frame: pd.DataFrame,
    input_path: Path,
    output_directory: Path,
    scenario: ScenarioSpec | None,
) -> AnalysisResult:
    frame = frame.sort_values(["step", "agent_id"]).reset_index(drop=True)
    per_step = _aggregate_trajectory(frame, scenario)
    per_step_path = output_directory / "per_step.csv"
    per_step.to_csv(per_step_path, index=False)

    summary = _trajectory_summary(frame, per_step, scenario)
    summary["source"] = str(input_path)
    summary_path = output_directory / "summary.json"
    _write_summary(summary_path, summary)

    generated: list[Path] = [summary_path, per_step_path]
    generated.append(_plot_trajectories(frame, output_directory, scenario))
    generated.append(_plot_swarm_dynamics(per_step, output_directory))
    for optional_path in (
        _plot_parameter_evolution(per_step, output_directory),
        _plot_correlations(frame, output_directory),
    ):
        if optional_path is not None:
            generated.append(optional_path)
    return AnalysisResult(
        input_path=input_path,
        output_directory=output_directory,
        kind="trajectory",
        scenario=scenario.key if scenario else None,
        generated_files=tuple(generated),
        summary=summary,
    )


def _plot_metric_group(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    title: str,
    output_path: Path,
) -> Path | None:
    if not columns:
        return None
    # 열 수에 따라 subplot을 늘려 서로 다른 단위의 선이 한 축에 겹치지 않게 한다.
    group_count = min(4, len(columns))
    buckets = [columns[index::group_count] for index in range(group_count)]
    figure, axes = plt.subplots(group_count, 1, figsize=(11.0, 2.5 * group_count), sharex=True)
    axes_array = np.atleast_1d(axes)
    if frame["step"].nunique() > 1:
        x_values = frame["step"]
        x_label = "environment step"
    else:
        # 평가 전용 로그는 학습 step을 증가시키지 않아 모든 행이 step=0일 수
        # 있다. 이때 수직선 대신 기록 순서를 표시하고 JSON에는 원래 step을
        # 그대로 남겨 데이터 의미를 보존한다.
        x_values = np.arange(len(frame))
        x_label = "logged record index (step is constant)"
    for axis, bucket in zip(axes_array, buckets):
        for column in bucket:
            axis.plot(x_values, frame[column], label=column, lw=1.45, marker="o", ms=3)
        axis.legend(loc="best", ncol=min(3, len(bucket)))
    axes_array[0].set_title(title)
    axes_array[-1].set_xlabel(x_label)
    return _save_figure(figure, output_path)


def _training_summary(frame: pd.DataFrame, scenario: ScenarioSpec | None) -> dict[str, object]:
    numeric = _numeric_columns(frame, exclude={"step"})
    summary: dict[str, object] = {
        "kind": "training",
        "scenario": scenario.key if scenario else None,
        "rows": int(len(frame)),
        "step_min": _finite_number(frame["step"].min()),
        "step_max": _finite_number(frame["step"].max()),
        "duplicate_steps": int(frame["step"].duplicated().sum()),
        "missing_numeric_values": int(frame[numeric].isna().sum().sum()) if numeric else 0,
        "metrics": {},
    }
    metrics: dict[str, object] = {}
    for column in numeric:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        metrics[column] = {
            "first": _finite_number(series.iloc[0]),
            "last": _finite_number(series.iloc[-1]),
            "min": _finite_number(series.min()),
            "max": _finite_number(series.max()),
            "mean": _finite_number(series.mean()),
        }
    summary["metrics"] = metrics
    return summary


def _analyze_training(
    frame: pd.DataFrame,
    input_path: Path,
    output_directory: Path,
    scenario: ScenarioSpec | None,
) -> AnalysisResult:
    frame = frame.sort_values("step").reset_index(drop=True)
    summary = _training_summary(frame, scenario)
    summary["source"] = str(input_path)
    summary_path = output_directory / "summary.json"
    _write_summary(summary_path, summary)
    generated: list[Path] = [summary_path]

    parameter_columns = _parameter_columns(frame)
    reward_columns = [
        column
        for column in _outcome_columns(frame)
        if "loss" not in column.lower() and column not in parameter_columns
    ]
    stability_tokens = ("loss", "entropy", "grad", "ratio", "variance", "pinn")
    stability_columns = [
        column
        for column in _numeric_columns(frame, exclude={"step"})
        if any(token in column.lower() for token in stability_tokens)
    ]
    diagnostic_columns = [
        column
        for column in _numeric_columns(frame, exclude={"step"})
        if column.startswith(("Count/", "Diagnostic/", "Diag/", "Constraint/"))
    ]

    plot_specs = (
        (reward_columns, "Learning and task outcomes", "learning_curves.png"),
        (parameter_columns, "Learned physical parameters", "physical_parameters.png"),
        (stability_columns, "Optimization and PINN stability", "training_stability.png"),
        (diagnostic_columns, "Task diagnostics", "task_diagnostics.png"),
    )
    for columns, title, filename in plot_specs:
        path = _plot_metric_group(
            frame,
            list(dict.fromkeys(columns)),
            title=title,
            output_path=output_directory / filename,
        )
        if path is not None:
            generated.append(path)
    correlation_path = _plot_correlations(frame, output_directory)
    if correlation_path is not None:
        generated.append(correlation_path)

    return AnalysisResult(
        input_path=input_path,
        output_directory=output_directory,
        kind="training",
        scenario=scenario.key if scenario else None,
        generated_files=tuple(generated),
        summary=summary,
    )


def analyze_csv(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    scenario: str = "auto",
) -> AnalysisResult:
    """하나의 PhySwarm CSV를 자동 판별해 분석 보고서를 생성한다."""

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"Input CSV is empty: {source}")
    if "step" not in frame:
        raise ValueError("Input CSV must contain a 'step' column")
    scenario_spec = _resolve_scenario(scenario, frame.columns)

    _configure_plot_style()
    if TRAJECTORY_REQUIRED_COLUMNS.issubset(frame.columns):
        return _analyze_trajectory(
            frame, source, destination, scenario_spec
        )
    return _analyze_training(frame, source, destination, scenario_spec)


def build_parser() -> argparse.ArgumentParser:
    """CLI parser를 별도 함수로 노출해 문서와 테스트에서 재사용한다."""

    parser = argparse.ArgumentParser(
        description="Generate reproducible PhySwarm CSV analysis plots and summaries."
    )
    parser.add_argument("--input", required=True, type=Path, help="trajectory or training CSV")
    parser.add_argument("--output", required=True, type=Path, help="directory for generated artifacts")
    parser.add_argument(
        "--scenario",
        default="auto",
        choices=("auto", *sorted(PAPER_SCENARIOS)),
        help="task semantics; inferred from columns by default",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 실행 후 생성 파일을 한 줄씩 출력한다."""

    arguments = build_parser().parse_args(argv)
    result = analyze_csv(
        arguments.input,
        arguments.output,
        scenario=arguments.scenario,
    )
    print(f"Analyzed {result.kind} CSV for scenario={result.scenario or 'unknown'}")
    for path in result.generated_files:
        print(path)
    return 0
