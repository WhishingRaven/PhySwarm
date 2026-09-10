"""논문 보충자료 S2.6의 평가 지표를 재사용 가능한 함수로 구현한다."""

from __future__ import annotations

import numpy as np


def control_smoothness(
    commands: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
) -> float:
    """식 (S174)의 robot-step 평균 L1 command variation ``J_smooth``.

    ``commands``는 ``(T,N,C)``이며 differential-drive 로그에서는 ``C=2``다.
    실패 로봇을 제외해야 할 때는 ``active_mask(T,N)``를 넘긴다. 이 경우 양쪽
    시점에 모두 활성인 robot-transition만 분모에 포함해, 죽은 로봇의 zero command가
    인위적인 jerk로 집계되지 않게 한다. mask가 없으면 논문의 ``N(T-1)`` 분모다.
    """

    values = np.asarray(commands, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] == 0:
        raise ValueError("commands must have shape (T, N, C) with C > 0")
    if not np.isfinite(values).all():
        raise ValueError("commands must contain only finite values")
    if values.shape[0] < 2 or values.shape[1] == 0:
        return 0.0

    variation = np.abs(values[1:] - values[:-1]).sum(axis=-1)
    if active_mask is None:
        return float(variation.sum() / variation.size)
    active = np.asarray(active_mask, dtype=np.bool_)
    if active.shape != values.shape[:2]:
        raise ValueError("active_mask must have shape (T, N)")
    valid_transition = active[1:] & active[:-1]
    if not valid_transition.any():
        return 0.0
    return float(variation[valid_transition].mean())


def collision_rate(collision_flags: np.ndarray) -> float:
    """식 (S175)의 robot-step 평균 충돌률을 반환한다."""

    flags = np.asarray(collision_flags, dtype=np.float64)
    if flags.size == 0:
        return 0.0
    return float(np.mean(flags > 0.0))


def foraging_efficiency(
    delivered_resources: float,
    num_robots: int,
    episode_length: float,
) -> float:
    """식 (S177)의 로봇당·시간당 배달 효율 ``Phi_ind``."""

    if delivered_resources < 0.0:
        raise ValueError("delivered_resources must be non-negative")
    if num_robots <= 0 or episode_length <= 0.0:
        raise ValueError("num_robots and episode_length must be positive")
    return float(delivered_resources / (num_robots * episode_length))


def transport_economy(
    trajectories: np.ndarray,
    delivered_resources: float,
    average_nest_resource_distance: float,
) -> float:
    """식 (S178)의 실제 이동거리 대비 유효 운송거리 비율.

    ``trajectories``는 ``(T, N, 2)``이며 NaN 위치는 실패/누락 로봇으로 보고
    해당 구간 이동거리에서 제외한다.
    """

    positions = np.asarray(trajectories, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("trajectories must have shape (T, N, 2)")
    if delivered_resources < 0.0 or average_nest_resource_distance < 0.0:
        raise ValueError("delivery count and average distance must be non-negative")
    if positions.shape[0] < 2:
        return 0.0

    displacements = positions[1:] - positions[:-1]
    valid = np.isfinite(displacements).all(axis=-1)
    traveled = np.where(valid, np.linalg.norm(displacements, axis=-1), 0.0).sum()
    if traveled <= 0.0:
        return 0.0
    return float(2.0 * delivered_resources * average_nest_resource_distance / traveled)


def formation_error(
    actual_positions: np.ndarray,
    target_positions: np.ndarray,
    *,
    allow_reflection: bool = False,
) -> float:
    """식 (S179)의 평행이동·회전 불변 RMS formation error.

    두 점 집합의 행 순서는 로봇과 목표점의 대응 관계다. 원형처럼 대응이
    모호한 경우 호출자가 각도 순서 등으로 먼저 정렬해야 한다. 기본값은
    ``SO(2)`` 회전만 허용하므로 거울 반사는 같은 formation으로 보지 않는다.
    """

    actual = np.asarray(actual_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    if actual.shape != target.shape or actual.ndim != 2 or actual.shape[1] != 2:
        raise ValueError("actual and target positions must share shape (N, 2)")
    if actual.shape[0] == 0:
        return 0.0

    actual_centered = actual - actual.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)

    # Orthogonal Procrustes: target @ R를 actual에 가장 가깝게 만드는 R.
    covariance = target_centered.T @ actual_centered
    left, _, right_transposed = np.linalg.svd(covariance)
    rotation = left @ right_transposed
    if not allow_reflection and np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transposed

    aligned_target = target_centered @ rotation
    squared_distance = np.sum((actual_centered - aligned_target) ** 2, axis=-1)
    return float(np.sqrt(np.mean(squared_distance)))


def mean_formation_error(errors: np.ndarray) -> float:
    """식 (S180)의 episode 시간 평균 formation error."""

    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("formation errors must be a one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("formation errors must contain only finite values")
    return float(values.mean()) if values.size else 0.0


def navigation_arrival_mask(
    trajectories: np.ndarray,
    goal_position: np.ndarray,
    arrival_threshold: float,
    *,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return the S181 centroid-arrival predicate for every timestep.

    A separate mask lets reporters distinguish a genuine arrival at ``T_max``
    from a failed episode that receives the same censored numeric value.
    """

    positions = np.asarray(trajectories, dtype=np.float64)
    goal = np.asarray(goal_position, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("trajectories must have shape (T, N, 2)")
    if goal.shape != (2,):
        raise ValueError("goal_position must have shape (2,)")
    if arrival_threshold <= 0.0:
        raise ValueError("arrival_threshold must be positive")

    if active_mask is None:
        active = np.ones(positions.shape[:2], dtype=np.bool_)
    else:
        active = np.asarray(active_mask, dtype=np.bool_)
        if active.shape != positions.shape[:2]:
            raise ValueError("active_mask must have shape (T, N)")

    arrivals = np.zeros(positions.shape[0], dtype=np.bool_)
    finite = np.isfinite(positions).all(axis=-1)
    for index in range(positions.shape[0]):
        valid = active[index] & finite[index]
        if valid.any():
            centroid = positions[index, valid].mean(axis=0)
            arrivals[index] = np.linalg.norm(centroid - goal) < arrival_threshold
    return arrivals


def navigation_time(
    trajectories: np.ndarray,
    goal_position: np.ndarray,
    arrival_threshold: float,
    *,
    steps: np.ndarray | None = None,
    max_time: float | None = None,
    active_mask: np.ndarray | None = None,
) -> float:
    """식 (S181)의 centroid 최초 도달 시각, 실패 시 ``T_max``.

    기본 step은 논문 표기처럼 1부터 시작한다. 실제 CSV step을 넘기면 그 값을
    그대로 반환하며, 도달하지 못한 episode는 ``max_time``(생략 시 마지막 step)을
    반환해 실패 run이 평균에서 탈락하지 않게 한다.
    """

    positions = np.asarray(trajectories, dtype=np.float64)
    arrivals = navigation_arrival_mask(
        positions,
        goal_position,
        arrival_threshold,
        active_mask=active_mask,
    )
    if steps is None:
        step_values = np.arange(1, positions.shape[0] + 1, dtype=np.float64)
    else:
        step_values = np.asarray(steps, dtype=np.float64)
        if step_values.shape != (positions.shape[0],):
            raise ValueError("steps must have shape (T,)")
        if not np.isfinite(step_values).all() or np.any(np.diff(step_values) < 0.0):
            raise ValueError("steps must be finite and non-decreasing")

    arrival_indices = np.flatnonzero(arrivals)
    if arrival_indices.size:
        return float(step_values[arrival_indices[0]])

    if max_time is not None:
        if not np.isfinite(max_time) or max_time < 0.0:
            raise ValueError("max_time must be finite and non-negative")
        return float(max_time)
    return float(step_values[-1]) if step_values.size else 0.0


def algebraic_connectivity(adjacency: np.ndarray) -> float:
    """무방향 weighted graph Laplacian의 두 번째 고유값 ``lambda_2``."""

    graph = np.asarray(adjacency, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if graph.shape[0] < 2:
        return 0.0
    if np.any(graph < 0.0) or not np.allclose(graph, graph.T, atol=1e-10):
        raise ValueError("adjacency must be symmetric and non-negative")

    graph = graph.copy()
    np.fill_diagonal(graph, 0.0)
    laplacian = np.diag(graph.sum(axis=1)) - graph
    eigenvalues = np.linalg.eigvalsh(laplacian)
    # 수치 오차로 생기는 -1e-16 정도의 값은 물리적으로 0이다.
    return float(max(eigenvalues[1], 0.0))


def relay_chain_connectivity(
    robot_positions: np.ndarray,
    base_position: np.ndarray,
    target_position: np.ndarray,
    communication_range: float,
    *,
    active_mask: np.ndarray | None = None,
) -> float:
    """식 (S182)의 base-target 연결 성분 ``lambda_2``를 계산한다.

    base와 target이 끊겨 있으면 논문 정의대로 즉시 0을 반환한다. 연결되어
    있으면 둘을 포함하는 connected component만 추출한 뒤 Fiedler 값을 구한다.
    통신 품질은 범위 경계에서 0, 같은 위치에서 1인 선형 거리 가중치로 둔다.
    """

    robots = np.asarray(robot_positions, dtype=np.float64)
    base = np.asarray(base_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if robots.ndim != 2 or robots.shape[1] != 2:
        raise ValueError("robot_positions must have shape (N, 2)")
    if base.shape != (2,) or target.shape != (2,):
        raise ValueError("base_position and target_position must have shape (2,)")
    if communication_range <= 0.0:
        raise ValueError("communication_range must be positive")

    if active_mask is None:
        active = np.ones(robots.shape[0], dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != (robots.shape[0],):
            raise ValueError("active_mask must have shape (N,)")
    nodes = np.vstack((robots[active], base, target))
    base_index = len(nodes) - 2
    target_index = len(nodes) - 1
    delta = nodes[:, None, :] - nodes[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    adjacency = np.where(
        (distances > 0.0) & (distances < communication_range),
        1.0 - distances / communication_range,
        0.0,
    )
    np.fill_diagonal(adjacency, 0.0)

    # BFS로 base가 속한 연결 성분을 찾고 target 포함 여부를 확인한다.
    visited = np.zeros(len(nodes), dtype=bool)
    queue = [base_index]
    visited[base_index] = True
    while queue:
        current = queue.pop(0)
        for neighbor in np.flatnonzero(adjacency[current] > 0.0):
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(int(neighbor))
    if not visited[target_index]:
        return 0.0

    connected_adjacency = adjacency[np.ix_(visited, visited)]
    return algebraic_connectivity(connected_adjacency)


def task_success_rate(connectivity_trials: np.ndarray) -> float:
    """식 (S183)--(S184): episode 중 ``lambda_2>0``인 trial의 비율."""

    values = np.asarray(connectivity_trials, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("connectivity_trials must have shape (trials, T)")
    if not np.isfinite(values).all():
        raise ValueError("connectivity trials must contain only finite values")
    if values.shape[0] == 0:
        return 0.0
    return float(np.mean(np.any(values > 0.0, axis=1)))


def time_to_connectivity(
    connectivity: np.ndarray,
    *,
    steps: np.ndarray | None = None,
    max_time: float | None = None,
) -> float:
    """식 (S185)의 최초 valid relay-chain 시각, 실패 시 ``T_max``."""

    values = np.asarray(connectivity, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("connectivity must be a one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("connectivity must contain only finite values")
    if steps is None:
        step_values = np.arange(1, values.size + 1, dtype=np.float64)
    else:
        step_values = np.asarray(steps, dtype=np.float64)
        if step_values.shape != values.shape:
            raise ValueError("steps must have the same shape as connectivity")
        if not np.isfinite(step_values).all() or np.any(np.diff(step_values) < 0.0):
            raise ValueError("steps must be finite and non-decreasing")
    connected = np.flatnonzero(values > 0.0)
    if connected.size:
        return float(step_values[connected[0]])
    if max_time is not None:
        if not np.isfinite(max_time) or max_time < 0.0:
            raise ValueError("max_time must be finite and non-negative")
        return float(max_time)
    return float(step_values[-1]) if step_values.size else 0.0
