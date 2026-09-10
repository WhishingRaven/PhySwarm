"""로봇 궤적에서 논문 S167--S173의 PINN loss를 실제로 조립한다.

기존 scenario trainer는 local observation에서 만든 경험적 target과 계수를
비교한다. 이 모듈은 그 surrogate를 사용하지 않는다. 고정 collocation grid에서
phase별 Gaussian KDE를 재구성하고, 두 시점의 유한차분으로 ``∂rho/∂t``를 만들며,
공간 flux divergence와 보존형 reaction source를 직접 계산한다.

Webots adapter가 side-channel로 제공하는 값은 명시적이다.

* 같은 좌표계의 robot position/phase/active mask trajectory
* 실행된 body-frame velocity와 agent 위치의 field basis (micro loss)
* arena frame의 위치와 agent 위치에서 평가한 task-specific field basis
* actor가 출력한 projected gain, diffusion, transition rate

모든 parameter interpolation과 residual 계산은 PyTorch graph 안에 있으므로 actor
parameter까지 gradient가 흐른다. 위치와 phase는 보통 rollout 상수지만, tensor가
gradient를 요구하는 경우 KDE도 미분 가능하다.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor

from learning.physics_loss import (
    PhysicsLossBreakdown,
    combine_physics_losses,
    micro_dynamics_loss,
    micro_edm_velocity_torch,
)
from physics.parameters import ParameterLayout
from tasks.specs import RuntimeMacroSpec, get_runtime_macro_spec, get_scenario


@dataclass(frozen=True)
class TorchDensityTrajectory:
    """정규 grid에서 평가한 phase KDE와 공간 미분.

    Shapes:
        density/laplacian: ``(B, T, S, H, W)``
        gradient: ``(B, T, S, H, W, 2)``
        agent_kernels: ``(B, T, N, H, W)``
    """

    density: Tensor
    gradient: Tensor
    laplacian: Tensor
    agent_kernels: Tensor


@dataclass(frozen=True)
class ExactMicroLoss:
    """실행 속도와 Micro-EDM 예측의 손실 및 진단 tensor."""

    loss: Tensor
    predicted_velocity: Tensor
    velocity_error: Tensor


@dataclass(frozen=True)
class ExactMacroLoss:
    """Macro-ADR loss와 각 residual 항을 분해한 결과."""

    loss: Tensor
    residual: Tensor
    density: Tensor
    density_time_derivative: Tensor
    advection_flux_divergence: Tensor
    diffusion_flux_divergence: Tensor
    reaction_source: Tensor


@dataclass(frozen=True)
class ExactPhysicsLoss:
    """선택한 exact mode의 합성 loss와 해석 가능한 원시 결과."""

    breakdown: PhysicsLossBreakdown
    micro_result: ExactMicroLoss | None
    macro_result: ExactMacroLoss | None


def _validate_grid(
    grid_points: Tensor,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    height, width = grid_shape
    if height < 3 or width < 3:
        raise ValueError("grid_shape must contain at least 3 points per axis")
    if grid_points.ndim != 2 or grid_points.shape != (height * width, 2):
        raise ValueError("grid_points must have shape (H*W, 2)")
    if not torch.isfinite(grid_points).all():
        raise ValueError("grid_points contains NaN or infinity")
    return height, width


def reconstruct_phase_density_torch(
    positions: Tensor,
    phases: Tensor,
    grid_points: Tensor,
    grid_shape: tuple[int, int],
    *,
    num_phases: int,
    bandwidth: float,
    active_mask: Tensor | None = None,
    population_size: int | None = None,
) -> TorchDensityTrajectory:
    """trajectory 전체의 S167 Gaussian KDE와 해석 공간미분을 계산한다.

    ``positions``는 ``(B,T,N,2)``, 정수 ``phases``는 ``(B,T,N)``다. phase 내부
    수가 아니라 초기 population으로 나누므로 각 phase density의 적분은 그
    phase의 population fraction이고 모든 phase 질량 합은 약 1이다.
    """

    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (B, T, N, 2)")
    if phases.shape != positions.shape[:-1]:
        raise ValueError("phases must have shape (B, T, N)")
    if num_phases <= 0:
        raise ValueError("num_phases must be positive")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    height, width = _validate_grid(grid_points, grid_shape)
    if not torch.isfinite(positions).all():
        raise ValueError("positions contains NaN or infinity")

    phase_index = phases.to(dtype=torch.long, device=positions.device)
    if torch.any((phase_index < 0) | (phase_index >= num_phases)):
        raise ValueError("phases contains an index outside [0, num_phases)")
    batch, time, robot_count, _ = positions.shape
    normalizer_population = robot_count if population_size is None else population_size
    if normalizer_population <= 0:
        raise ValueError("population_size must be positive")

    if active_mask is None:
        active = torch.ones(
            (batch, time, robot_count),
            dtype=positions.dtype,
            device=positions.device,
        )
    else:
        expected = positions.shape[:-1]
        if active_mask.shape not in {expected, expected + (1,)}:
            raise ValueError("active_mask must have shape (B,T,N) or (B,T,N,1)")
        active = active_mask.squeeze(-1) if active_mask.ndim == 4 else active_mask
        active = active.to(dtype=positions.dtype, device=positions.device)

    points = grid_points.to(dtype=positions.dtype, device=positions.device)
    delta = points[None, None, None, :, :] - positions[:, :, :, None, :]
    distance_squared = torch.sum(delta.square(), dim=-1)
    h2 = bandwidth * bandwidth
    gaussian_constant = 1.0 / (2.0 * torch.pi)
    kernel = gaussian_constant * torch.exp(-0.5 * distance_squared / h2)
    kernel = kernel / (normalizer_population * bandwidth**2)
    kernel = kernel * active[..., None]

    one_hot = functional.one_hot(phase_index, num_classes=num_phases).to(positions.dtype)
    density_flat = torch.einsum("btnp,btns->btsp", kernel, one_hot)
    kernel_gradient = -(delta / h2) * kernel[..., None]
    gradient_flat = torch.einsum("btnpd,btns->btspd", kernel_gradient, one_hot)
    kernel_laplacian = (distance_squared / (h2 * h2) - 2.0 / h2) * kernel
    laplacian_flat = torch.einsum("btnp,btns->btsp", kernel_laplacian, one_hot)

    return TorchDensityTrajectory(
        density=density_flat.reshape(batch, time, num_phases, height, width),
        gradient=gradient_flat.reshape(
            batch, time, num_phases, height, width, 2
        ),
        laplacian=laplacian_flat.reshape(
            batch, time, num_phases, height, width
        ),
        agent_kernels=kernel.reshape(batch, time, robot_count, height, width),
    )


def interpolate_phase_agent_values(
    agent_values: Tensor,
    density: TorchDensityTrajectory,
    phases: Tensor,
    *,
    num_phases: int,
    active_mask: Tensor | None = None,
    epsilon: float = 1.0e-8,
) -> Tensor:
    """agent parameter를 phase별 kernel regression으로 grid에 보간한다.

    입력은 ``(B,T,N,C)``이고 출력은 ``(B,T,S,H,W,C)``다. 어느 phase에도
    robot/kernel support가 없는 cell은 NaN 대신 0으로 둔다. KDE의 공통
    normalization은 분자와 분모에서 상쇄된다.
    """

    if agent_values.ndim != 4:
        raise ValueError("agent_values must have shape (B,T,N,C)")
    if phases.shape != agent_values.shape[:-1]:
        raise ValueError("phases must have shape (B,T,N)")
    if density.agent_kernels.shape[:3] != agent_values.shape[:3]:
        raise ValueError("density kernels and agent_values must share B,T,N axes")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    phase_index = phases.to(dtype=torch.long, device=agent_values.device)
    one_hot = functional.one_hot(phase_index, num_classes=num_phases).to(
        agent_values.dtype
    )
    kernels = density.agent_kernels.to(
        dtype=agent_values.dtype, device=agent_values.device
    )
    if active_mask is not None:
        active = active_mask.squeeze(-1) if active_mask.ndim == 4 else active_mask
        kernels = kernels * active.to(
            dtype=agent_values.dtype, device=agent_values.device
        )[..., None, None]

    numerator = torch.einsum(
        "btnhw,btns,btnc->btshwc", kernels, one_hot, agent_values
    )
    denominator = torch.einsum("btnhw,btns->btshw", kernels, one_hot)
    return torch.where(
        denominator[..., None] > epsilon,
        numerator / denominator[..., None].clamp_min(epsilon),
        torch.zeros_like(numerator),
    )


def _finite_difference(values: Tensor, *, dim: int, spacing: float) -> Tensor:
    """중앙차분과 양 끝 one-sided 차분을 gradient-safe하게 계산한다."""

    if spacing <= 0.0:
        raise ValueError("grid spacing must be positive")
    moved = values.movedim(dim, -1)
    if moved.shape[-1] < 3:
        raise ValueError("finite difference axis must contain at least 3 points")
    left = (moved[..., 1] - moved[..., 0]).unsqueeze(-1) / spacing
    center = (moved[..., 2:] - moved[..., :-2]) / (2.0 * spacing)
    right = (moved[..., -1] - moved[..., -2]).unsqueeze(-1) / spacing
    return torch.cat((left, center, right), dim=-1).movedim(-1, dim)


def spatial_divergence_2d(
    vector_field: Tensor,
    *,
    x_spacing: float,
    y_spacing: float,
) -> Tensor:
    """``(...,H,W,2)`` vector field의 ``dFx/dx + dFy/dy``를 구한다."""

    if vector_field.ndim < 3 or vector_field.shape[-1] != 2:
        raise ValueError("vector_field must have shape (...,H,W,2)")
    dfx_dx = _finite_difference(
        vector_field[..., 0], dim=-1, spacing=x_spacing
    )
    dfy_dy = _finite_difference(
        vector_field[..., 1], dim=-2, spacing=y_spacing
    )
    return dfx_dx + dfy_dy


def _regular_grid_spacing(
    grid_points: Tensor,
    grid_shape: tuple[int, int],
) -> tuple[float, float]:
    """flattened grid가 uniform xy mesh인지 검증하고 ``dx,dy``를 반환한다."""

    height, width = grid_shape
    grid = grid_points.reshape(height, width, 2)
    x_values = grid[0, :, 0]
    y_values = grid[:, 0, 1]
    dx = x_values[1:] - x_values[:-1]
    dy = y_values[1:] - y_values[:-1]
    reference_x = x_values[None, :].expand(height, -1)
    reference_y = y_values[:, None].expand(-1, width)
    regular = (
        torch.allclose(grid[..., 0], reference_x)
        and torch.allclose(grid[..., 1], reference_y)
        and torch.allclose(dx, torch.full_like(dx, dx[0]))
        and torch.allclose(dy, torch.full_like(dy, dy[0]))
    )
    if not regular or dx[0] <= 0.0 or dy[0] <= 0.0:
        raise ValueError("grid_points must be an increasing regular xy grid")
    return float(dx[0].detach().cpu()), float(dy[0].detach().cpu())


def exact_micro_edm_loss(
    executed_velocity: Tensor,
    field_bases_at_agents: Tensor,
    advection_gains: Tensor,
    diffusion: Tensor,
    density_at_agents: Tensor,
    density_gradient_at_agents: Tensor,
    *,
    active_mask: Tensor | None = None,
    epsilon: float = 0.05,
) -> ExactMicroLoss:
    """S168--S170의 executed-velocity Micro-EDM loss를 직접 계산한다."""

    predicted = micro_edm_velocity_torch(
        field_bases_at_agents,
        advection_gains,
        diffusion,
        density_at_agents,
        density_gradient_at_agents,
        epsilon=epsilon,
    )
    loss = micro_dynamics_loss(
        executed_velocity,
        predicted,
        active_mask=active_mask,
    )
    return ExactMicroLoss(
        loss=loss,
        predicted_velocity=predicted,
        velocity_error=executed_velocity - predicted,
    )


def exact_micro_loss_from_context(
    context: Mapping[str, object],
    named_parameters: Mapping[str, Tensor],
    layout: ParameterLayout,
    *,
    active_mask: Tensor | None = None,
    epsilon: float = 0.05,
) -> ExactMicroLoss:
    """legacy PPO mini-batch side-channel을 canonical exact micro loss로 변환한다.

    context의 배열은 ``(B,T,N,...)``이고 actor parameter는 흔히
    ``(B*T*N,1)``로 평탄화되어 있다. 이 함수가 shape를 검증하고 다시 정렬하므로
    세 scenario trainer가 같은 reshape 규칙을 복제하지 않는다.
    """

    required = {
        "executed_velocity",
        "field_bases",
        "density",
        "density_gradient",
        "active_mask",
    }
    missing = required.difference(context)
    if missing:
        raise ValueError(f"physics context is missing fields: {sorted(missing)}")
    first_gain_name = f"w_{layout.advection_names[0]}"
    if first_gain_name not in named_parameters or "k_diff" not in named_parameters:
        raise ValueError("named_parameters does not match the ParameterLayout")
    reference = named_parameters[first_gain_name]
    device, dtype = reference.device, reference.dtype

    bases = torch.as_tensor(context["field_bases"], dtype=dtype, device=device)
    if bases.ndim != 5 or bases.shape[-2:] != (len(layout.advection_names), 2):
        raise ValueError("field_bases must have shape (B,T,N,K,2) matching layout")
    batch, time, agents = bases.shape[:3]
    expected_scalars = batch * time * agents

    gain_channels = []
    for name in layout.advection_names:
        key = f"w_{name}"
        if key not in named_parameters:
            raise ValueError(f"named_parameters is missing {key!r}")
        values = named_parameters[key]
        if values.numel() != expected_scalars:
            raise ValueError(f"parameter {key!r} does not align with physics context")
        gain_channels.append(values.reshape(batch, time, agents, 1))
    gains = torch.cat(gain_channels, dim=-1)

    diffusion_values = named_parameters["k_diff"]
    if diffusion_values.numel() != expected_scalars:
        raise ValueError("k_diff does not align with physics context")
    diffusion = diffusion_values.reshape(batch, time, agents, 1)
    executed = torch.as_tensor(
        context["executed_velocity"], dtype=dtype, device=device
    )
    density = torch.as_tensor(context["density"], dtype=dtype, device=device)
    gradient = torch.as_tensor(
        context["density_gradient"], dtype=dtype, device=device
    )
    context_active = torch.as_tensor(
        context["active_mask"], dtype=torch.bool, device=device
    )
    if executed.shape != (batch, time, agents, 2):
        raise ValueError("executed_velocity does not align with field_bases")
    if density.shape != (batch, time, agents):
        raise ValueError("density does not align with field_bases")
    if gradient.shape != (batch, time, agents, 2):
        raise ValueError("density_gradient does not align with field_bases")
    if context_active.shape != (batch, time, agents):
        raise ValueError("active_mask does not align with field_bases")
    if active_mask is not None:
        policy_active = active_mask.squeeze(-1) if active_mask.ndim == 4 else active_mask
        try:
            context_active = context_active & torch.broadcast_to(
                policy_active.to(dtype=torch.bool, device=device), context_active.shape
            )
        except RuntimeError as error:
            raise ValueError("policy active_mask does not align with physics context") from error

    return exact_micro_edm_loss(
        executed,
        bases,
        gains,
        diffusion,
        density,
        gradient,
        active_mask=context_active,
        epsilon=epsilon,
    )


def _reshape_parameter_channel(
    named_parameters: Mapping[str, Tensor],
    name: str,
    *,
    batch: int,
    time: int,
    agents: int,
) -> Tensor:
    """평탄화된 actor channel을 side-channel의 ``B,T,N,1``로 복원한다."""

    if name not in named_parameters:
        raise ValueError(f"named_parameters is missing {name!r}")
    values = named_parameters[name]
    if values.numel() != batch * time * agents:
        raise ValueError(f"parameter {name!r} does not align with physics context")
    return values.reshape(batch, time, agents, 1)


def _runtime_grid(
    spec: RuntimeMacroSpec,
    *,
    dtype: torch.dtype,
    device: torch.device,
    grid_shape: tuple[int, int] | None,
) -> tuple[Tensor, tuple[int, int]]:
    """arena local 좌표계의 고정 xy collocation grid를 만든다."""

    height, width = spec.grid_shape if grid_shape is None else grid_shape
    if height < 3 or width < 3:
        raise ValueError("macro grid must contain at least three points per axis")
    arena_width, arena_height = spec.arena_size
    x = torch.linspace(
        -arena_width / 2.0,
        arena_width / 2.0,
        width,
        dtype=dtype,
        device=device,
    )
    y = torch.linspace(
        -arena_height / 2.0,
        arena_height / 2.0,
        height,
        dtype=dtype,
        device=device,
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1), (
        height,
        width,
    )


def _macro_state_trajectory(
    context: Mapping[str, object],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """transition payload를 중복 없는 ``T+1`` state trajectory로 조립한다."""

    required = {
        "position_start",
        "position_end",
        "phase_start",
        "phase_end",
        "active_start",
        "active_end",
        "field_bases_global",
        "reaction_gates",
    }
    missing = required.difference(context)
    if missing:
        raise ValueError(f"physics context is missing fields: {sorted(missing)}")

    start_position = torch.as_tensor(
        context["position_start"], dtype=dtype, device=device
    )
    end_position = torch.as_tensor(
        context["position_end"], dtype=dtype, device=device
    )
    start_phase = torch.as_tensor(
        context["phase_start"], dtype=torch.long, device=device
    )
    end_phase = torch.as_tensor(
        context["phase_end"], dtype=torch.long, device=device
    )
    start_active = torch.as_tensor(
        context["active_start"], dtype=torch.bool, device=device
    )
    end_active = torch.as_tensor(
        context["active_end"], dtype=torch.bool, device=device
    )
    if start_position.ndim != 4 or start_position.shape[-1] != 2:
        raise ValueError("position_start must have shape (B,T,N,2)")
    if end_position.shape != start_position.shape:
        raise ValueError("position_end must match position_start")
    state_prefix = start_position.shape[:-1]
    for name, value in (
        ("phase_start", start_phase),
        ("phase_end", end_phase),
        ("active_start", start_active),
        ("active_end", end_active),
    ):
        if value.shape != state_prefix:
            raise ValueError(f"{name} must have shape (B,T,N)")

    # 인접 transition의 end/start는 같은 Webots state여야 한다. 활성 robot만
    # 검사해 dead-agent placeholder 좌표는 허용하되 episode/reset 혼입은 잡는다.
    if start_position.shape[1] > 1:
        shared_active = end_active[:, :-1] & start_active[:, 1:]
        position_delta = torch.abs(end_position[:, :-1] - start_position[:, 1:])
        if torch.any(position_delta[shared_active] > 1.0e-5):
            raise ValueError("physics transitions do not form a contiguous trajectory")
        if torch.any(
            (end_phase[:, :-1] != start_phase[:, 1:]) & shared_active
        ):
            raise ValueError("phase transitions do not form a contiguous trajectory")

    positions = torch.cat((start_position[:, :1], end_position), dim=1)
    phases = torch.cat((start_phase[:, :1], end_phase), dim=1)
    active = torch.cat((start_active[:, :1], end_active), dim=1)
    return positions, phases, active, start_active


def _effective_reaction_rates(
    context: Mapping[str, object],
    named_parameters: Mapping[str, Tensor],
    layout: ParameterLayout,
    spec: RuntimeMacroSpec,
    phases_start: Tensor,
    phases_end: Tensor,
    active_start: Tensor,
    active_end: Tensor,
    *,
    dt: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, tuple[tuple[int, int], ...]]:
    """학습률 ``lambda*chi``와 관측된 event 전이를 한 보존 graph로 만든다.

    learned edge는 actor tensor를 그대로 사용하므로 macro residual에서 gradient가
    흐른다. NPC 출력이 없는 event edge는 실제 phase change indicator를 ``dt``로
    나눈 경험적 rate로 사용한다. 이는 센서/기하 gate를 별도 신경망으로 꾸며내지
    않고 실제 simulator 사건을 보존식에 포함하는 가장 직접적인 추정치다.
    """

    batch, time, agents = phases_start.shape
    if len(layout.reaction_names) != len(spec.learned_reaction_edges):
        raise ValueError("reaction layout and runtime learned edges disagree")
    gates = torch.as_tensor(context["reaction_gates"], dtype=dtype, device=device)
    expected_gate_shape = (batch, time, agents, len(layout.reaction_names))
    if gates.shape != expected_gate_shape:
        raise ValueError(
            f"reaction_gates must have shape {expected_gate_shape}, got {tuple(gates.shape)}"
        )
    if torch.any((gates < 0.0) | (gates > 1.0)):
        raise ValueError("reaction_gates must lie in [0,1]")

    learned_channels: list[Tensor] = []
    for index, name in enumerate(layout.reaction_names):
        rate = _reshape_parameter_channel(
            named_parameters,
            f"lambda_{name}",
            batch=batch,
            time=time,
            agents=agents,
        ).squeeze(-1)
        # 논문 식 (7)의 effective rate lambda_mn * chi_mn.
        learned_channels.append(rate * gates[..., index])

    # 한 actor rate가 여러 source phase의 같은 task event를 조절할 수 있다.
    # Foraging의 pickup은 approach->home과 trail->home 양쪽에 작동하므로
    # edge별 channel을 복제하되 원 actor tensor와 gradient는 공유한다.
    channels = list(learned_channels)
    edges = list(spec.learned_reaction_edges)
    for channel, source, destination in spec.additional_learned_reaction_edges:
        channels.append(learned_channels[channel])
        edges.append((source, destination))

    valid_transition = active_start & active_end
    for source, destination in spec.resolved_event_reaction_edges:
        happened = (
            (phases_start == source)
            & (phases_end == destination)
            & valid_transition
        )
        channels.append(happened.to(dtype=dtype) / dt)
        edges.append((source, destination))

    if channels:
        return torch.stack(channels, dim=-1), tuple(edges)
    return torch.empty(
        (batch, time, agents, 0), dtype=dtype, device=device
    ), ()


def exact_macro_loss_from_context(
    context: Mapping[str, object],
    named_parameters: Mapping[str, Tensor],
    layout: ParameterLayout,
    scenario: str,
    *,
    dt: float,
    phase_contract: str = "legacy",
    bandwidth: float = 0.1,
    grid_shape: tuple[int, int] | None = None,
) -> ExactMacroLoss:
    """PPO side-channel에서 실제 trajectory 기반 Macro-ADR loss를 계산한다.

    task field는 action 시점의 agent sample을 phase별 Gaussian kernel regression으로
    collocation grid에 보간한다. 따라서 세 과제마다 별도의 grid-field 코드를
    복제하지 않으며 stochastic exploration field도 실행 당시 값을 그대로 쓴다.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if not layout.advection_names:
        raise ValueError("layout requires at least one advection field")
    first_name = f"w_{layout.advection_names[0]}"
    if first_name not in named_parameters:
        raise ValueError("named_parameters does not match the ParameterLayout")
    reference = named_parameters[first_name]
    dtype, device = reference.dtype, reference.device
    spec = get_runtime_macro_spec(scenario, phase_contract=phase_contract)

    positions, phases, active, active_start = _macro_state_trajectory(
        context, dtype=dtype, device=device
    )
    batch, state_time, agents = phases.shape
    time = state_time - 1
    if torch.any(phases >= len(spec.phase_names)):
        raise ValueError("physics context phase exceeds runtime scenario phase count")

    gains = torch.cat(
        [
            _reshape_parameter_channel(
                named_parameters,
                f"w_{name}",
                batch=batch,
                time=time,
                agents=agents,
            )
            for name in layout.advection_names
        ],
        dim=-1,
    )
    diffusion = _reshape_parameter_channel(
        named_parameters,
        "k_diff",
        batch=batch,
        time=time,
        agents=agents,
    )
    bases_at_agents = torch.as_tensor(
        context["field_bases_global"], dtype=dtype, device=device
    )
    expected_bases = (batch, time, agents, len(layout.advection_names), 2)
    if bases_at_agents.shape != expected_bases:
        raise ValueError(
            f"field_bases_global must have shape {expected_bases}, "
            f"got {tuple(bases_at_agents.shape)}"
        )

    grid_points, resolved_grid_shape = _runtime_grid(
        spec, dtype=dtype, device=device, grid_shape=grid_shape
    )
    density_at_parameters = reconstruct_phase_density_torch(
        positions[:, :-1],
        phases[:, :-1],
        grid_points,
        resolved_grid_shape,
        num_phases=len(spec.phase_names),
        bandwidth=bandwidth,
        active_mask=active_start,
        population_size=agents,
    )
    flat_bases = bases_at_agents.reshape(batch, time, agents, -1)
    grid_bases = interpolate_phase_agent_values(
        flat_bases,
        density_at_parameters,
        phases[:, :-1],
        num_phases=len(spec.phase_names),
        active_mask=active_start,
    ).reshape(
        batch,
        time,
        len(spec.phase_names),
        resolved_grid_shape[0] * resolved_grid_shape[1],
        len(layout.advection_names),
        2,
    )
    reaction_rates, reaction_edges = _effective_reaction_rates(
        context,
        named_parameters,
        layout,
        spec,
        phases[:, :-1],
        phases[:, 1:],
        active[:, :-1],
        active[:, 1:],
        dt=dt,
        dtype=dtype,
        device=device,
    )
    return exact_macro_adr_loss(
        positions,
        phases,
        grid_points,
        resolved_grid_shape,
        grid_bases,
        gains,
        diffusion,
        reaction_rates,
        reaction_edges,
        num_phases=len(spec.phase_names),
        bandwidth=bandwidth,
        dt=dt,
        active_mask=active,
        population_size=agents,
    )


def exact_physics_loss_from_context(
    context: Mapping[str, object],
    named_parameters: Mapping[str, Tensor],
    layout: ParameterLayout,
    scenario: str,
    *,
    mode: str,
    dt: float,
    phase_contract: str = "legacy",
    macro_weight: float = 1.0,
    bandwidth: float = 0.1,
    grid_shape: tuple[int, int] | None = None,
    active_mask: Tensor | None = None,
) -> ExactPhysicsLoss:
    """``exact_micro``, ``exact_macro``, ``exact_joint`` ablation을 한곳에서 조립한다."""

    valid_modes = {"exact_micro", "exact_macro", "exact_joint"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")
    first_name = f"w_{layout.advection_names[0]}"
    reference = named_parameters[first_name]
    zero = reference.sum() * 0.0

    resolved_context: Mapping[str, object] = context
    if get_scenario(scenario).key == "navigation":
        resolved_context = _navigation_parameterized_context(
            context, named_parameters, layout
        )

    micro_result = None
    macro_result = None
    micro_loss = zero
    macro_loss = zero
    if mode in {"exact_micro", "exact_joint"}:
        micro_result = exact_micro_loss_from_context(
            resolved_context,
            named_parameters,
            layout,
            active_mask=active_mask,
        )
        micro_loss = micro_result.loss
    if mode in {"exact_macro", "exact_joint"}:
        macro_result = exact_macro_loss_from_context(
            resolved_context,
            named_parameters,
            layout,
            scenario,
            dt=dt,
            phase_contract=phase_contract,
            bandwidth=bandwidth,
            grid_shape=grid_shape,
        )
        macro_loss = macro_result.loss
    return ExactPhysicsLoss(
        breakdown=combine_physics_losses(
            micro_loss,
            macro_loss,
            macro_weight=macro_weight,
        ),
        micro_result=micro_result,
        macro_result=macro_result,
    )


def _navigation_parameterized_context(
    context: Mapping[str, object],
    named_parameters: Mapping[str, Tensor],
    layout: ParameterLayout,
) -> Mapping[str, object]:
    """Navigation shape basis를 현재 actor ``beta``로 autograd 안에서 재계산한다.

    rollout에는 실행 당시 sample action으로 만든 basis도 있지만 PPO update는
    현재 distribution mean의 parameter를 미분한다. 저장된 basis를 그대로 쓰면
    beta에는 physics gradient가 전혀 흐르지 않는다. Supervisor가 저장한 최소
    centroid geometry로 global/body basis를 모두 다시 만들어 이 끊김을 막는다.
    """

    required = {
        "field_bases",
        "field_bases_global",
        "navigation_shape_geometry",
        "heading_start",
    }
    missing = required.difference(context)
    if missing:
        raise ValueError(
            f"navigation physics context is missing fields: {sorted(missing)}"
        )
    if "shape" not in layout.advection_names or "beta" not in named_parameters:
        raise ValueError("navigation layout requires shape and beta parameters")
    shape_index = layout.advection_names.index("shape")
    reference = named_parameters["beta"]
    dtype, device = reference.dtype, reference.device
    original_body = torch.as_tensor(context["field_bases"], dtype=dtype, device=device)
    original_global = torch.as_tensor(
        context["field_bases_global"], dtype=dtype, device=device
    )
    if original_body.shape != original_global.shape or original_body.ndim != 5:
        raise ValueError("navigation field bases must have shape (B,T,N,K,2)")
    batch, time, agents = original_body.shape[:3]
    geometry = torch.as_tensor(
        context["navigation_shape_geometry"], dtype=dtype, device=device
    )
    headings = torch.as_tensor(
        context["heading_start"], dtype=dtype, device=device
    )
    if geometry.shape != (batch, time, agents, 3):
        raise ValueError("navigation_shape_geometry must have shape (B,T,N,3)")
    if headings.shape != (batch, time, agents):
        raise ValueError("heading_start must have shape (B,T,N)")
    beta = _reshape_parameter_channel(
        named_parameters,
        "beta",
        batch=batch,
        time=time,
        agents=agents,
    ).squeeze(-1)
    x_relative, y_relative, target_radius = geometry.unbind(dim=-1)
    warped_distance = torch.sqrt(
        (x_relative.square() + beta * y_relative.square()).clamp_min(1.0e-12)
    )
    radial_error = warped_distance - target_radius
    scale = 50.0 * radial_error / warped_distance.clamp_min(1.0e-6)
    shape_global = torch.stack(
        (scale * x_relative, scale * beta * y_relative), dim=-1
    )
    cosine, sine = torch.cos(headings), torch.sin(headings)
    shape_body = torch.stack(
        (
            shape_global[..., 0] * cosine + shape_global[..., 1] * sine,
            -shape_global[..., 0] * sine + shape_global[..., 1] * cosine,
        ),
        dim=-1,
    )

    def replace_shape(original: Tensor, replacement: Tensor) -> Tensor:
        before = original[..., :shape_index, :]
        after = original[..., shape_index + 1 :, :]
        return torch.cat((before, replacement.unsqueeze(-2), after), dim=-2)

    resolved = dict(context)
    resolved["field_bases"] = replace_shape(original_body, shape_body)
    resolved["field_bases_global"] = replace_shape(original_global, shape_global)
    return resolved


def _reaction_source_from_edges(
    density: Tensor,
    rate_fields: Tensor,
    transition_edges: Sequence[tuple[int, int]],
) -> Tensor:
    """source-phase rate field로 질량 보존 reaction source를 구성한다."""

    reaction = torch.zeros_like(density)
    for edge_index, (source, destination) in enumerate(transition_edges):
        if source == destination:
            raise ValueError("reaction transitions must connect distinct phases")
        if not (0 <= source < density.shape[2] and 0 <= destination < density.shape[2]):
            raise ValueError("reaction transition contains an invalid phase index")
        # rate_fields는 모든 phase 보간값을 포함하지만 source phase robot의 값을
        # 사용해야 lambda_source,destination의 의미가 유지된다.
        flow = rate_fields[:, :, source, ..., edge_index] * density[:, :, source]
        stoichiometry = torch.zeros(
            density.shape[2], dtype=density.dtype, device=density.device
        )
        stoichiometry[source] = -1.0
        stoichiometry[destination] = 1.0
        reaction = reaction + flow.unsqueeze(2) * stoichiometry.view(1, 1, -1, 1, 1)
    return reaction


def exact_macro_adr_loss(
    positions: Tensor,
    phases: Tensor,
    grid_points: Tensor,
    grid_shape: tuple[int, int],
    field_bases_on_grid: Tensor,
    advection_gains: Tensor,
    diffusion: Tensor,
    transition_rates: Tensor,
    transition_edges: Sequence[tuple[int, int]],
    *,
    num_phases: int,
    bandwidth: float,
    dt: float,
    active_mask: Tensor | None = None,
    collocation_mask: Tensor | None = None,
    population_size: int | None = None,
    interior_only: bool = True,
) -> ExactMacroLoss:
    """trajectory로부터 S167, S171--S173의 macro loss를 end-to-end 계산한다.

    ``field_bases_on_grid``는 phase 공통 ``(B,T,P,K,2)`` 또는 phase별
    ``(B,T,S,P,K,2)``다. projected agent parameter는 ``advection_gains``
    ``(B,T,N,K)``, ``diffusion`` ``(B,T,N,1)``, ``transition_rates``
    ``(B,T,N,E)`` 형태다. 각 edge tuple은 ``(source_phase, destination_phase)``다.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    height, width = _validate_grid(grid_points, grid_shape)
    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (B,T+1,N,2)")
    batch, state_time, robot_count, _ = positions.shape
    if advection_gains.ndim != 4:
        raise ValueError("advection_gains must have shape (B,T,N,K)")
    parameter_shape = advection_gains.shape[:3]
    if parameter_shape != (batch, state_time - 1, robot_count):
        raise ValueError(
            "positions must contain exactly one more state than parameter timesteps"
        )
    if phases.shape != positions.shape[:-1]:
        raise ValueError("phases must have shape (B,T+1,N)")
    if diffusion.shape != parameter_shape + (1,):
        raise ValueError("diffusion must have shape (B,T,N,1)")
    if transition_rates.shape != parameter_shape + (len(transition_edges),):
        raise ValueError("transition_rates must have shape (B,T,N,E)")
    if advection_gains.shape[-1] <= 0:
        raise ValueError("at least one advection field is required")
    if not torch.isfinite(advection_gains).all():
        raise ValueError("advection_gains contains NaN or infinity")
    if not torch.isfinite(diffusion).all() or torch.any(diffusion < 0.0):
        raise ValueError("diffusion must be finite and non-negative")
    if not torch.isfinite(transition_rates).all() or torch.any(transition_rates < 0.0):
        raise ValueError("transition_rates must be finite and non-negative")

    density_result = reconstruct_phase_density_torch(
        positions,
        phases,
        grid_points,
        grid_shape,
        num_phases=num_phases,
        bandwidth=bandwidth,
        active_mask=active_mask,
        population_size=population_size,
    )
    # parameter/action t는 state t에서 실행되어 state t+1을 만든다. KDE는 T+1개
    # 상태 모두에서 만들고, 공간 flux와 reaction에는 처음 T개만 사용한다.
    parameter_density = TorchDensityTrajectory(
        density=density_result.density[:, :-1],
        gradient=density_result.gradient[:, :-1],
        laplacian=density_result.laplacian[:, :-1],
        agent_kernels=density_result.agent_kernels[:, :-1],
    )
    parameter_phases = phases[:, :-1]
    parameter_active = active_mask[:, :-1] if active_mask is not None else None
    gain_field = interpolate_phase_agent_values(
        advection_gains,
        parameter_density,
        parameter_phases,
        num_phases=num_phases,
        active_mask=parameter_active,
    )
    diffusion_field = interpolate_phase_agent_values(
        diffusion,
        parameter_density,
        parameter_phases,
        num_phases=num_phases,
        active_mask=parameter_active,
    ).squeeze(-1)
    rate_fields = interpolate_phase_agent_values(
        transition_rates,
        parameter_density,
        parameter_phases,
        num_phases=num_phases,
        active_mask=parameter_active,
    )

    time = advection_gains.shape[1]
    point_count = height * width
    field_count = advection_gains.shape[-1]
    if field_bases_on_grid.shape == (batch, time, point_count, field_count, 2):
        bases = field_bases_on_grid.reshape(
            batch, time, 1, height, width, field_count, 2
        ).expand(-1, -1, num_phases, -1, -1, -1, -1)
    elif field_bases_on_grid.shape == (
        batch,
        time,
        num_phases,
        point_count,
        field_count,
        2,
    ):
        bases = field_bases_on_grid.reshape(
            batch, time, num_phases, height, width, field_count, 2
        )
    else:
        raise ValueError(
            "field_bases_on_grid must have shape (B,T,P,K,2) or (B,T,S,P,K,2)"
        )
    bases = bases.to(dtype=advection_gains.dtype, device=advection_gains.device)
    if not torch.isfinite(bases).all():
        raise ValueError("field_bases_on_grid contains NaN or infinity")

    advection_velocity = torch.sum(gain_field[..., None] * bases, dim=-2)
    advection_flux = advection_velocity * parameter_density.density[..., None]
    diffusion_flux = diffusion_field[..., None] * parameter_density.gradient

    # regular_grid는 meshgrid(indexing='xy')이므로 x는 W, y는 H 축이다.
    x_spacing, y_spacing = _regular_grid_spacing(
        grid_points.to(dtype=positions.dtype, device=positions.device),
        grid_shape,
    )

    advection_divergence = spatial_divergence_2d(
        advection_flux,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
    )
    diffusion_divergence = spatial_divergence_2d(
        diffusion_flux,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
    )
    reaction = _reaction_source_from_edges(
        parameter_density.density,
        rate_fields,
        transition_edges,
    )

    density_dt = (density_result.density[:, 1:] - density_result.density[:, :-1]) / dt
    residual = (
        density_dt
        + advection_divergence
        - diffusion_divergence
        - reaction
    )

    mask = torch.ones_like(residual, dtype=torch.bool)
    if interior_only:
        mask[..., 0, :] = False
        mask[..., -1, :] = False
        mask[..., :, 0] = False
        mask[..., :, -1] = False
    if collocation_mask is not None:
        try:
            mask = mask & torch.broadcast_to(
                collocation_mask.to(dtype=torch.bool, device=mask.device),
                residual.shape,
            )
        except RuntimeError as error:
            raise ValueError("collocation_mask is not broadcastable to residual") from error
    selected = residual[mask]
    loss = selected.square().mean() if selected.numel() else residual.sum() * 0.0

    return ExactMacroLoss(
        loss=loss,
        residual=residual,
        density=density_result.density,
        density_time_derivative=density_dt,
        advection_flux_divergence=advection_divergence,
        diffusion_flux_divergence=diffusion_divergence,
        reaction_source=reaction,
    )
