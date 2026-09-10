"""로봇 입자 위치를 Macro-ADR 위상 밀도로 재구성한다.

논문 식 (11)과 보충 식 (S167)은 각 로봇을 Gaussian kernel로 펼쳐 연속
밀도를 만든다. 이 모듈은 시뮬레이터나 신경망에 의존하지 않는 NumPy 구현을
제공하므로, 학습 데이터와 실제 로봇 로그에도 같은 정의를 적용할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DensityEstimate:
    """collocation point에서 평가한 위상별 KDE와 공간 미분."""

    density: np.ndarray
    gradient: np.ndarray
    laplacian: np.ndarray


def reconstruct_phase_density(
    positions: np.ndarray,
    phases: np.ndarray,
    sample_points: np.ndarray,
    *,
    num_phases: int,
    bandwidth: float,
    population_size: int | None = None,
) -> DensityEstimate:
    """Gaussian KDE로 위상 밀도와 gradient/laplacian을 계산한다.

    Args:
        positions: 현재 로봇 위치 ``(N, d)``. 이 프로젝트에서는 ``d=2``다.
        phases: 각 로봇의 정수 위상 번호 ``(N,)``.
        sample_points: 밀도를 평가할 collocation point ``(C, d)``.
        num_phases: Macro-ADR 위상 수 ``M``.
        bandwidth: Gaussian kernel 폭 ``h``. 미터 단위 위치에는 미터 단위다.
        population_size: 식 (11)의 정규화에 쓸 전체 로봇 수. 죽은 로봇을
            제외한 좌표만 전달해도 초기 개체수 기준 질량을 유지하려면 지정한다.

    Returns:
        ``density``는 ``(M, C)``, ``gradient``는 ``(M, C, d)``,
        ``laplacian``은 ``(M, C)`` 모양이다.

    이 구현은 각 위상 내부 개체수로 나누지 않고 전체 ``N``으로 나눈다.
    따라서 위상 밀도의 공간 적분은 해당 위상 로봇 비율에 대응하며, 모든
    위상 질량의 합은 경계 절단 오차를 제외하면 1이다.
    """

    robot_positions = np.asarray(positions, dtype=np.float64)
    robot_phases = np.asarray(phases)
    points = np.asarray(sample_points, dtype=np.float64)

    if robot_positions.ndim != 2:
        raise ValueError("positions must have shape (N, d)")
    if points.ndim != 2 or points.shape[1] != robot_positions.shape[1]:
        raise ValueError("sample_points must have shape (C, d) matching positions")
    if robot_phases.shape != (robot_positions.shape[0],):
        raise ValueError("phases must have shape (N,)")
    if num_phases <= 0:
        raise ValueError("num_phases must be positive")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if np.any((robot_phases < 0) | (robot_phases >= num_phases)):
        raise ValueError("phases contains an index outside [0, num_phases)")

    n_robots, dimension = robot_positions.shape
    normalizer_population = n_robots if population_size is None else population_size
    if normalizer_population <= 0:
        raise ValueError("population_size must be positive")

    density = np.zeros((num_phases, points.shape[0]), dtype=np.float64)
    gradient = np.zeros(
        (num_phases, points.shape[0], dimension), dtype=np.float64
    )
    laplacian = np.zeros((num_phases, points.shape[0]), dtype=np.float64)

    if n_robots == 0:
        return DensityEstimate(density=density, gradient=gradient, laplacian=laplacian)

    # delta[c, i]는 로봇 i에서 collocation point c로 향하는 벡터다.
    delta = points[:, None, :] - robot_positions[None, :, :]
    distance_squared = np.sum(delta * delta, axis=-1)
    h2 = bandwidth * bandwidth

    gaussian_constant = (2.0 * np.pi) ** (-dimension / 2.0)
    kernel = gaussian_constant * np.exp(-0.5 * distance_squared / h2)
    kernel /= normalizer_population * bandwidth**dimension

    # Gaussian 미분을 해석적으로 계산해 수치 차분 잡음을 피한다.
    kernel_gradient = -(delta / h2) * kernel[..., None]
    kernel_laplacian = (
        distance_squared / (h2 * h2) - dimension / h2
    ) * kernel

    for phase_index in range(num_phases):
        phase_mask = robot_phases == phase_index
        if not np.any(phase_mask):
            continue
        density[phase_index] = kernel[:, phase_mask].sum(axis=1)
        gradient[phase_index] = kernel_gradient[:, phase_mask].sum(axis=1)
        laplacian[phase_index] = kernel_laplacian[:, phase_mask].sum(axis=1)

    return DensityEstimate(
        density=density,
        gradient=gradient,
        laplacian=laplacian,
    )


def regular_grid(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    *,
    resolution: int | tuple[int, int] = 64,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """밀도 평가용 2차원 균일 grid와 셀 면적을 만든다.

    반환되는 점은 ``(C, 2)``로 평탄화되어 있으며, ``grid_shape``으로 다시
    reshape할 수 있다. 셀 면적은 밀도 적분과 E_ADR 계산에 사용한다.
    """

    if isinstance(resolution, int):
        nx = ny = resolution
    else:
        nx, ny = resolution
    if nx < 2 or ny < 2:
        raise ValueError("resolution must contain at least two points per axis")

    x_min, x_max = x_limits
    y_min, y_max = y_limits
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("grid limits must be strictly increasing")

    x_values = np.linspace(x_min, x_max, nx)
    y_values = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    cell_area = (x_values[1] - x_values[0]) * (y_values[1] - y_values[0])
    return points, (ny, nx), float(cell_area)
