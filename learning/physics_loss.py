"""gradient를 보존하는 논문식 RL-PINN 구성요소.

과제별 MAPPO 전략은 일부 서로 다른 경험적 surrogate를 갖는다. 이 모듈은
논문 식 (S166)--(S173)을 그대로 조립할 수 있는 공통 연산을
제공한다. 환경 adapter는 task-specific field와 반응 source만 준비하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from physics.parameters import ParameterLayout


@dataclass(frozen=True)
class TorchProjectedParameters:
    """PyTorch graph 안에서 물리 매니폴드로 투영된 NPC 출력."""

    omega: Tensor
    advection_gains: Tensor
    diffusion: Tensor
    reaction_rates: Tensor
    extras: dict[str, Tensor]

    def as_named_dict(self, layout: ParameterLayout) -> dict[str, Tensor]:
        """기존 trainer가 기대하는 이름별 tensor 딕셔너리를 반환한다."""

        values: dict[str, Tensor] = {
            f"w_{name}": self.advection_gains[..., index : index + 1]
            for index, name in enumerate(layout.advection_names)
        }
        values["k_diff"] = self.diffusion
        values.update(
            {
                f"lambda_{name}": self.reaction_rates[..., index : index + 1]
                for index, name in enumerate(layout.reaction_names)
            }
        )
        values.update(self.extras)
        values["advection_weights"] = self.omega
        return values


@dataclass(frozen=True)
class PhysicsLossBreakdown:
    """총 PINN 손실과 해석 가능한 두 구성 항."""

    total: Tensor
    micro: Tensor
    macro: Tensor


def ridge_least_squares_2row(
    design: Tensor,
    target: Tensor,
    *,
    regularization: float = 1.0e-4,
) -> Tensor:
    """Solve a batched two-row ridge problem without backend decompositions.

    ``design`` has shape ``(..., 2, K)`` and ``target`` has shape
    ``(..., 2, 1)``.  The returned ``(..., K, 1)`` tensor is mathematically
    equivalent to ``solve(A.T @ A + lambda I, A.T @ y)``.  Using the dual
    two-dimensional system avoids the MPS decomposition kernel that can abort
    on the rank-deficient ``2 x K`` systems used by the legacy PINN targets.

    The 2x2 determinant is written with the Cauchy--Binet identity.  This avoids
    subtracting two large, nearly equal Gram products when the two rows are
    almost parallel, while retaining gradients through both inputs.
    """

    if design.ndim < 2 or design.shape[-2] != 2 or design.shape[-1] == 0:
        raise ValueError("design must have shape (..., 2, K) with K > 0")
    expected_target_shape = design.shape[:-1] + (1,)
    if target.shape != expected_target_shape:
        raise ValueError(
            f"target must have shape {tuple(expected_target_shape)}, "
            f"received {tuple(target.shape)}"
        )
    if design.device != target.device or design.dtype != target.dtype:
        raise ValueError("design and target must share device and dtype")
    if not design.is_floating_point() or not target.is_floating_point():
        raise TypeError("design and target must be floating-point tensors")
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")

    row_x = design[..., 0, :]
    row_y = design[..., 1, :]
    norm_x = row_x.square().sum(dim=-1)
    norm_y = row_y.square().sum(dim=-1)
    cross_inner = (row_x * row_y).sum(dim=-1)

    # Sum of squared 2x2 minors equals
    # ||row_x||^2 ||row_y||^2 - <row_x,row_y>^2, without cancellation.
    minors = (
        row_x.unsqueeze(-1) * row_y.unsqueeze(-2)
        - row_y.unsqueeze(-1) * row_x.unsqueeze(-2)
    )
    minor_square_sum = 0.5 * minors.square().sum(dim=(-2, -1))
    determinant = (
        minor_square_sum
        + regularization * (norm_x + norm_y)
        + regularization**2
    )

    target_x = target[..., 0, 0]
    target_y = target[..., 1, 0]
    dual_x = (
        (norm_y + regularization) * target_x - cross_inner * target_y
    ) / determinant
    dual_y = (
        (norm_x + regularization) * target_y - cross_inner * target_x
    ) / determinant

    solution = row_x * dual_x.unsqueeze(-1) + row_y * dual_y.unsqueeze(-1)
    return solution.unsqueeze(-1)


def detached_target_least_squares(design: Tensor, target: Tensor) -> Tensor:
    """Fit a non-differentiable supervision target on a stable backend.

    MPS decomposition kernels can abort for batched rank-deficient systems.
    Historical target fitting does not require gradients, so MPS inputs are
    deliberately solved on CPU and copied back. CPU inputs stay on CPU.
    """

    if design.ndim < 2 or target.shape[:-2] != design.shape[:-2]:
        raise ValueError("design and target must have matching batch dimensions")
    if design.shape[-2] != target.shape[-2] or target.shape[-1] != 1:
        raise ValueError("expected design (..., M, K) and target (..., M, 1)")
    if design.device != target.device or design.dtype != target.dtype:
        raise ValueError("design and target must share device and dtype")
    with torch.no_grad():
        solve_design = design.detach().cpu() if design.device.type == "mps" else design.detach()
        solve_target = target.detach().cpu() if target.device.type == "mps" else target.detach()
        solution = torch.linalg.lstsq(solve_design, solve_target).solution
    return solution.to(device=design.device)


def project_torch_parameters(
    raw_actions: Tensor,
    layout: ParameterLayout,
) -> TorchProjectedParameters:
    """논문 식 (22)의 매니폴드 투영을 gradient 손실 없이 적용한다."""

    if raw_actions.ndim < 1 or raw_actions.shape[-1] != layout.action_dim:
        actual = raw_actions.shape[-1] if raw_actions.ndim else 0
        raise ValueError(
            f"Expected action_dim={layout.action_dim}, received {actual}"
        )
    if not torch.isfinite(raw_actions).all():
        raise ValueError("raw_actions contains NaN or infinity")

    cursor = 0
    n_advection = len(layout.advection_names)
    omega = torch.softmax(
        raw_actions[..., cursor : cursor + n_advection]
        * layout.advection_temperature,
        dim=-1,
    )
    if layout.sparsity_threshold > 0.0:
        suppressed = torch.relu(omega - layout.sparsity_threshold)
        normalizer = suppressed.sum(dim=-1, keepdim=True)
        normalized = suppressed / normalizer.clamp_min(1.0e-8)
        omega = torch.where(normalizer > 1.0e-8, normalized, omega)
    cursor += n_advection

    d_min, d_max = layout.diffusion_bounds
    bounded = raw_actions[..., cursor : cursor + 1]
    if layout.bounded_transform == "sigmoid":
        diffusion_unit = torch.sigmoid(bounded)
    else:
        diffusion_unit = (torch.clamp(bounded, -1.0, 1.0) + 1.0) / 2.0
    diffusion = d_min + (d_max - d_min) * diffusion_unit
    cursor += 1

    n_reaction = len(layout.reaction_names)
    reaction_latent = raw_actions[..., cursor : cursor + n_reaction]
    if layout.bounded_transform == "sigmoid":
        reaction_unit = torch.sigmoid(reaction_latent)
    else:
        reaction_unit = (torch.clamp(reaction_latent, -1.0, 1.0) + 1.0) / 2.0
    reaction_rates = layout.reaction_max * reaction_unit
    cursor += n_reaction

    extras: dict[str, Tensor] = {}
    for name, (lower, upper) in layout.extra_bounds.items():
        extra_latent = raw_actions[..., cursor : cursor + 1]
        if layout.bounded_transform == "sigmoid":
            unit_value = torch.sigmoid(extra_latent)
        else:
            unit_value = (torch.clamp(extra_latent, -1.0, 1.0) + 1.0) / 2.0
        extras[name] = lower + (upper - lower) * unit_value
        cursor += 1

    return TorchProjectedParameters(
        omega=omega,
        advection_gains=omega * layout.advection_scale,
        diffusion=diffusion,
        reaction_rates=reaction_rates,
        extras=extras,
    )


def _broadcast_scalar_field(values: Tensor, vector: Tensor) -> Tensor:
    """``(...,)`` 또는 ``(...,1)`` scalar field를 vector 축과 맞춘다."""

    if values.ndim == vector.ndim - 1:
        return values.unsqueeze(-1)
    return values


def micro_edm_velocity_torch(
    field_bases: Tensor,
    advection_weights: Tensor,
    diffusion: Tensor,
    density: Tensor,
    density_gradient: Tensor,
    *,
    epsilon: float = 0.05,
) -> Tensor:
    """논문 식 (S168),(S169)의 differentiable Micro-EDM 속도."""

    if field_bases.ndim < 2 or advection_weights.shape != field_bases.shape[:-1]:
        raise ValueError("advection_weights must match field_bases without vector axis")
    if density_gradient.shape != field_bases.shape[:-2] + (field_bases.shape[-1],):
        raise ValueError("density_gradient must have shape (..., d)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    advection = torch.sum(advection_weights.unsqueeze(-1) * field_bases, dim=-2)
    diffusion = _broadcast_scalar_field(diffusion, density_gradient)
    density = _broadcast_scalar_field(density, density_gradient)
    return advection - diffusion * density_gradient / (density + epsilon)


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    """padding/dead-agent mask를 적용하되 빈 batch에서도 유한한 0을 돌려준다."""

    if mask is None:
        return values.mean()
    broadcast_mask = torch.broadcast_to(mask.to(values.dtype), values.shape)
    denominator = broadcast_mask.sum()
    if denominator.detach().item() == 0:
        # ``values``와 연결된 0을 반환해야 backward()가 안전하다.
        return values.sum() * 0.0
    return (values * broadcast_mask).sum() / denominator


def micro_dynamics_loss(
    executed_velocity: Tensor,
    edm_velocity: Tensor,
    *,
    active_mask: Tensor | None = None,
) -> Tensor:
    """논문 식 (26)/(S170)의 평균 제곱 Micro-EDM 일관성 손실."""

    if executed_velocity.shape != edm_velocity.shape:
        raise ValueError("executed_velocity and edm_velocity must have identical shape")
    squared_error = torch.sum((executed_velocity - edm_velocity) ** 2, dim=-1)
    if active_mask is not None and active_mask.ndim == squared_error.ndim + 1:
        active_mask = active_mask.squeeze(-1)
    return _masked_mean(squared_error, active_mask)


def macro_adr_residual_torch(
    density_time_derivative: Tensor,
    advection_flux_divergence: Tensor,
    diffusion_flux_divergence: Tensor,
    reaction_source: Tensor,
) -> Tensor:
    """논문 식 (25)/(S172)의 differentiable 점별 잔차."""

    reference_shape = density_time_derivative.shape
    others = (
        advection_flux_divergence,
        diffusion_flux_divergence,
        reaction_source,
    )
    if any(term.shape != reference_shape for term in others):
        raise ValueError("all Macro-ADR residual terms must have identical shape")
    return (
        density_time_derivative
        + advection_flux_divergence
        - diffusion_flux_divergence
        - reaction_source
    )


def macro_adr_loss(
    density_time_derivative: Tensor,
    advection_flux_divergence: Tensor,
    diffusion_flux_divergence: Tensor,
    reaction_source: Tensor,
    *,
    collocation_mask: Tensor | None = None,
) -> Tensor:
    """모든 위상/collocation point에 대한 논문식 ``L_ADR`` 평균."""

    residual = macro_adr_residual_torch(
        density_time_derivative,
        advection_flux_divergence,
        diffusion_flux_divergence,
        reaction_source,
    )
    return _masked_mean(residual.square(), collocation_mask)


def combine_physics_losses(
    micro_loss: Tensor,
    macro_loss: Tensor,
    *,
    macro_weight: float = 1.0,
) -> PhysicsLossBreakdown:
    """논문 S166의 ``L_PINN = L_dyn + beta L_ADR``를 구성한다."""

    if macro_weight < 0.0:
        raise ValueError("macro_weight must be non-negative")
    total = micro_loss + macro_weight * macro_loss
    return PhysicsLossBreakdown(total=total, micro=micro_loss, macro=macro_loss)


def soft_cap_nonnegative_loss(loss: Tensor, *, maximum: float) -> Tensor:
    """큰 physics residual을 유한하게 제한하면서 gradient를 완전히 끊지 않는다.

    ``torch.clamp(loss, max=...)``는 임계값을 넘는 순간 미분이 0이 되어, 특히
    초기 Macro-ADR residual이 큰 실험에서 actor가 물리 오차를 줄일 신호를 잃는다.
    ``M L / (M + L)``은 ``[0,M)``에 있고 모든 유한 ``L>=0``에서 양의 기울기를
    유지한다. 음수 loss는 구현 오류이므로 조용히 보정하지 않고 거부한다.
    """

    if maximum <= 0.0:
        raise ValueError("maximum must be positive")
    if torch.any(loss.detach() < 0.0):
        raise ValueError("loss must be non-negative")
    return maximum * loss / (maximum + loss)
