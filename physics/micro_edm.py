"""Macro-ADR flux를 로봇 속도와 차동구동 명령으로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MicroEDMVelocity:
    """해석을 위해 advection/diffusion/합성 속도를 함께 보관한다."""

    advection: np.ndarray
    diffusion_compensation: np.ndarray
    desired: np.ndarray


def micro_edm_velocity(
    field_bases: np.ndarray,
    advection_weights: np.ndarray,
    diffusion: np.ndarray,
    density: np.ndarray,
    density_gradient: np.ndarray,
    *,
    epsilon: float = 0.05,
) -> MicroEDMVelocity:
    """논문 식 (5),(6),(18)의 Micro-EDM 속도를 계산한다.

    ``field_bases``는 ``(..., K, d)``, ``advection_weights``는 ``(..., K)``,
    ``density_gradient``는 ``(..., d)`` 모양이다. ``diffusion``과 ``density``는
    앞쪽 batch 모양이 같거나 NumPy broadcast가 가능해야 한다.

    여기서 장 기저는 이미 ``b_k=-grad Phi_k``인 속도 방향이다. 따라서 함수
    내부에서 부호를 다시 뒤집지 않는다.
    """

    bases = np.asarray(field_bases, dtype=np.float64)
    weights = np.asarray(advection_weights, dtype=np.float64)
    coefficient = np.asarray(diffusion, dtype=np.float64)
    local_density = np.asarray(density, dtype=np.float64)
    gradient = np.asarray(density_gradient, dtype=np.float64)

    if bases.ndim < 2 or weights.shape != bases.shape[:-1]:
        raise ValueError("weights must match field_bases without its vector axis")
    if gradient.shape != bases.shape[:-2] + (bases.shape[-1],):
        raise ValueError("density_gradient must have shape (..., d)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if np.any(coefficient < 0.0) or np.any(local_density < 0.0):
        raise ValueError("diffusion and density must be non-negative")

    advection = np.sum(weights[..., None] * bases, axis=-2)
    # 마지막 벡터 축과 broadcast되도록 scalar 계수에 축을 하나 붙인다.
    scale = coefficient / (local_density + epsilon)
    diffusion_velocity = -np.asarray(scale)[..., None] * gradient
    return MicroEDMVelocity(
        advection=advection,
        diffusion_compensation=diffusion_velocity,
        desired=advection + diffusion_velocity,
    )


def differential_drive_wheel_speeds(
    desired_body_velocity: np.ndarray,
    *,
    wheel_radius: float = 0.02,
    wheel_separation: float = 0.05685,
    heading_gain: float = 2.0,
    max_linear_speed: float = 0.12,
    max_angular_speed: float = 4.0,
    max_wheel_speed: float = 6.28,
) -> np.ndarray:
    """body-frame 목표 속도를 E-puck 좌/우 바퀴 각속도로 바꾼다.

    논문 식 (19)의 heading tracker와 포화 조건을 따른다. 반환 배열의 마지막
    축은 ``[left, right]``다. 목표가 옆이나 뒤를 향할 때는 먼저 회전하도록
    전진 속도에 ``cos(angle)`` 기반 정렬 계수를 적용한다.
    """

    velocity = np.asarray(desired_body_velocity, dtype=np.float64)
    if velocity.ndim < 1 or velocity.shape[-1] != 2:
        raise ValueError("desired_body_velocity must have shape (..., 2)")
    positive_constants = (
        wheel_radius,
        wheel_separation,
        heading_gain,
        max_linear_speed,
        max_angular_speed,
        max_wheel_speed,
    )
    if any(value <= 0.0 for value in positive_constants):
        raise ValueError("drive geometry, gains and limits must be positive")

    target_angle = np.arctan2(velocity[..., 1], velocity[..., 0])
    angular_command = np.clip(
        heading_gain * target_angle, -max_angular_speed, max_angular_speed
    )
    alignment = np.clip(np.cos(target_angle), 0.0, 1.0)
    linear_command = np.minimum(
        np.linalg.norm(velocity, axis=-1), max_linear_speed
    ) * alignment

    left_linear = linear_command - 0.5 * wheel_separation * angular_command
    right_linear = linear_command + 0.5 * wheel_separation * angular_command
    wheel_speeds = np.stack(
        (left_linear / wheel_radius, right_linear / wheel_radius), axis=-1
    )
    return np.clip(wheel_speeds, -max_wheel_speed, max_wheel_speed)
