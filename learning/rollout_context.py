"""PPO observation과 분리된 physics side-channel의 수집·저장 계약.

정확한 PINN에 필요한 위치, 실행 속도, field basis를 actor observation에 붙이면
기존 checkpoint가 깨지고 actor가 원래 허용되지 않은 전역 정보를 보게 된다.
이 모듈은 그 데이터를 학습 전용 side-channel로 저장한다. legacy PPO buffer가
세 시나리오에 복제되어 있어도 검증/정렬 규칙은 여기 한 곳에서 공유한다.

좌표계는 일부러 둘로 나눈다. ``field_bases``와 ``executed_velocity``는 실제
Micro-EDM 명령과 동일한 robot body frame이고, ``field_bases_global``과
``position_*``은 공간 미분을 계산하는 Macro-ADR용 arena frame이다. 두 값을
혼용하면 수식은 실행돼도 회전한 로봇의 flux 방향이 틀리므로 builder 단계에서
shape와 유한성을 엄격히 검사한다.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence

import numpy as np


MICRO_CONTEXT_KEYS = frozenset(
    {
        "executed_velocity",
        "field_bases",
        "density",
        "density_gradient",
        "active_mask",
    }
)

MACRO_CONTEXT_KEYS = frozenset(
    {
        "field_bases_global",
        "position_start",
        "position_end",
        "phase_start",
        "phase_end",
        "active_start",
        "active_end",
        "reaction_gates",
    }
)


def body_to_world_vectors(vectors: np.ndarray, headings: np.ndarray) -> np.ndarray:
    """body-frame 2-D vector 묶음을 동일 로봇의 arena frame으로 회전한다.

    ``vectors``는 ``(..., K, 2)`` 또는 ``(..., 2)``, ``headings``는 vector
    basis 축을 제외한 robot prefix ``(...)``다. Webots yaw가 반시계 방향
    양수이므로 표준 2-D 회전행렬 ``R(theta)``를 사용한다.
    """

    body = np.asarray(vectors, dtype=np.float32)
    yaw = np.asarray(headings, dtype=np.float32)
    if body.ndim < 2 or body.shape[-1] != 2:
        raise ValueError("vectors must end with a 2-D coordinate axis")
    if body.shape[:-2] == yaw.shape:
        yaw = yaw[..., None]
    elif body.shape[:-1] != yaw.shape:
        raise ValueError("headings must match the vector robot axes")
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    x_body, y_body = body[..., 0], body[..., 1]
    return np.stack(
        (cosine * x_body - sine * y_body, sine * x_body + cosine * y_body),
        axis=-1,
    ).astype(np.float32)


def phase_indices_from_one_hot(
    indicators: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """one-hot phase 관측을 정수 index로 바꾸고 잘못된 활성 행을 거부한다.

    죽은 robot의 관측은 모두 0일 수 있으므로 ``active_mask=False``인 행만
    예외로 허용하고 index 0으로 채운다. 활성 robot에 phase가 없거나 둘 이상
    켜져 있으면 KDE 질량이 잘못된 phase로 들어가기 전에 즉시 실패한다.
    """

    values = np.asarray(indicators, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 2:
        raise ValueError("phase indicators must have shape (E,N,S), S >= 2")
    active = (
        np.ones(values.shape[:2], dtype=np.bool_)
        if active_mask is None
        else np.asarray(active_mask, dtype=np.bool_)
    )
    if active.shape != values.shape[:2]:
        raise ValueError("active_mask must match phase indicator E,N axes")
    is_hot = values > 0.5
    if np.any(is_hot.sum(axis=-1)[active] != 1):
        raise ValueError("each active robot must have exactly one phase")
    phases = np.argmax(values, axis=-1).astype(np.int64)
    phases[~active] = 0
    return phases


def build_physics_transition(
    *,
    executed_velocity: np.ndarray,
    field_bases: np.ndarray,
    density: np.ndarray,
    density_gradient: np.ndarray,
    active_mask: np.ndarray,
    field_bases_global: np.ndarray,
    position_start: np.ndarray,
    position_end: np.ndarray,
    phase_start: np.ndarray,
    phase_end: np.ndarray,
    active_start: np.ndarray,
    active_end: np.ndarray,
    reaction_gates: np.ndarray,
    auxiliary_fields: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """한 control interval의 exact micro+macro payload를 안전하게 복사한다.

    모든 배열의 첫 두 축은 ``(parallel environments, robots)``다. position과
    phase를 start/end로 명시해 PPO transition ``t -> t+1``의 시간 정렬을
    보존한다. ``reaction_gates``의 마지막 축은 ``ParameterLayout``의 reaction
    channel 순서이며 reaction이 없는 과제는 길이 0인 축을 전달한다.
    """

    payload = build_micro_transition(
        executed_velocity=executed_velocity,
        field_bases=field_bases,
        density=density,
        density_gradient=density_gradient,
        active_mask=active_mask,
    )
    prefix = payload["executed_velocity"].shape[:2]
    global_bases = np.asarray(field_bases_global, dtype=np.float32)
    start_position = np.asarray(position_start, dtype=np.float32)
    end_position = np.asarray(position_end, dtype=np.float32)
    start_phase = np.asarray(phase_start, dtype=np.int64)
    end_phase = np.asarray(phase_end, dtype=np.int64)
    start_active = np.asarray(active_start, dtype=np.bool_)
    end_active = np.asarray(active_end, dtype=np.bool_)
    gates = np.asarray(reaction_gates, dtype=np.float32)

    if global_bases.shape != payload["field_bases"].shape:
        raise ValueError("field_bases_global must match field_bases shape")
    if start_position.shape != prefix + (2,) or end_position.shape != prefix + (2,):
        raise ValueError("position_start/end must have shape (E,N,2)")
    if start_phase.shape != prefix or end_phase.shape != prefix:
        raise ValueError("phase_start/end must have shape (E,N)")
    if start_active.shape != prefix or end_active.shape != prefix:
        raise ValueError("active_start/end must have shape (E,N)")
    if gates.ndim != 3 or gates.shape[:2] != prefix:
        raise ValueError("reaction_gates must have shape (E,N,R)")
    numeric = (global_bases, start_position, end_position, gates)
    if any(not np.isfinite(value).all() for value in numeric):
        raise ValueError("macro physics transition contains NaN or infinity")
    if np.any(start_phase < 0) or np.any(end_phase < 0):
        raise ValueError("phase indices must be non-negative")

    payload.update(
        {
            "field_bases_global": global_bases.copy(),
            "position_start": start_position.copy(),
            "position_end": end_position.copy(),
            "phase_start": start_phase.copy(),
            "phase_end": end_phase.copy(),
            "active_start": start_active.copy(),
            "active_end": end_active.copy(),
            "reaction_gates": gates.copy(),
        }
    )
    # 일부 field basis 자체가 actor parameter에 의존한다. Navigation의 beta처럼
    # trainer에서 basis를 다시 미분 가능하게 계산할 최소 기하량만 여기에 싣는다.
    # 이름을 열어 두되 E,N prefix와 유한성은 공통 계약으로 강제한다.
    for name, value in (auxiliary_fields or {}).items():
        if name in payload:
            raise ValueError(f"auxiliary physics field {name!r} collides with core data")
        array = np.asarray(value)
        if array.ndim < 2 or array.shape[:2] != prefix:
            raise ValueError(
                f"auxiliary physics field {name!r} must start with (E,N)"
            )
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"auxiliary physics field {name!r} is not finite")
        payload[name] = array.copy()
    return payload


def build_micro_transition(
    *,
    executed_velocity: np.ndarray,
    field_bases: np.ndarray,
    density: np.ndarray,
    density_gradient: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """환경의 서로 다른 내부 배열을 canonical ``(E,N,...)`` payload로 복사한다.

    field basis의 순서는 해당 `ParameterLayout.advection_names`와 같아야 한다.
    반환 배열을 복사하므로 다음 Webots step에서 환경 buffer가 갱신되어도 이미
    기록한 transition이 바뀌지 않는다.
    """

    velocity = np.asarray(executed_velocity, dtype=np.float32)
    bases = np.asarray(field_bases, dtype=np.float32)
    rho = np.asarray(density, dtype=np.float32)
    gradient = np.asarray(density_gradient, dtype=np.float32)
    active = np.asarray(active_mask, dtype=np.bool_)
    if velocity.ndim != 3 or velocity.shape[-1] != 2:
        raise ValueError("executed_velocity must have shape (E,N,2)")
    prefix = velocity.shape[:2]
    if bases.ndim != 4 or bases.shape[:2] != prefix or bases.shape[-1] != 2:
        raise ValueError("field_bases must have shape (E,N,K,2)")
    if rho.shape != prefix:
        raise ValueError("density must have shape (E,N)")
    if gradient.shape != prefix + (2,):
        raise ValueError("density_gradient must have shape (E,N,2)")
    if active.shape != prefix:
        raise ValueError("active_mask must have shape (E,N)")
    values = (velocity, bases, rho, gradient)
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("micro physics transition contains NaN or infinity")
    return {
        "executed_velocity": velocity.copy(),
        "field_bases": bases.copy(),
        "density": rho.copy(),
        "density_gradient": gradient.copy(),
        "active_mask": active.copy(),
    }


def record_physics_transition(
    episode_context: MutableMapping[str, np.ndarray],
    transition: Mapping[str, np.ndarray],
    *,
    step: int,
    episode_length: int,
    num_envs: int,
    num_agents: int,
) -> None:
    """환경의 한 step physics payload를 time-first episode 배열에 기록한다.

    모든 값은 ``(E,N,...)``이어야 한다. key/shape가 episode 중간에 바뀌면 즉시
    실패한다. 조용한 broadcasting은 서로 다른 agent 데이터를 섞을 수 있기
    때문이다.
    """

    if not 0 <= step < episode_length:
        raise IndexError("physics transition step is outside the episode")
    if not transition:
        return
    for name, value in transition.items():
        array = np.asarray(value)
        if array.ndim < 2 or array.shape[:2] != (num_envs, num_agents):
            raise ValueError(
                f"physics field {name!r} must start with (num_envs,num_agents)"
            )
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"physics field {name!r} contains NaN or infinity")
        if name not in episode_context:
            episode_context[name] = np.zeros(
                (episode_length, *array.shape), dtype=array.dtype
            )
        target = episode_context[name]
        if target.shape != (episode_length, *array.shape) or target.dtype != array.dtype:
            raise ValueError(f"physics field {name!r} changed shape or dtype mid-episode")
        target[step] = array


class PhysicsContextBuffer:
    """episode physics payload를 PPO circular-buffer index와 함께 보관한다."""

    def __init__(self, *, episode_length: int, buffer_size: int, num_agents: int):
        if min(episode_length, buffer_size, num_agents) <= 0:
            raise ValueError("physics buffer dimensions must be positive")
        self.episode_length = episode_length
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.data: dict[str, np.ndarray] = {}

    def insert(
        self,
        context: Mapping[str, np.ndarray] | None,
        episode_indices: np.ndarray,
    ) -> None:
        """``(T,E,N,...)`` payload를 PPO buffer의 episode slot에 복사한다."""

        if context is None:
            if self.data:
                raise ValueError("physics context disappeared after buffer initialization")
            return
        if not context:
            raise ValueError("physics context must not be an empty mapping")
        if self.data and set(context) != set(self.data):
            raise ValueError("physics context keys changed between episodes")

        indices = np.asarray(episode_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any((indices < 0) | (indices >= self.buffer_size)):
            raise ValueError("episode_indices contains an invalid buffer slot")
        for name, value in context.items():
            array = np.asarray(value)
            expected_prefix = (
                self.episode_length,
                len(indices),
                self.num_agents,
            )
            if array.ndim < 3 or array.shape[:3] != expected_prefix:
                raise ValueError(
                    f"physics field {name!r} must have shape (T,E,N,...)"
                )
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise ValueError(f"physics field {name!r} contains NaN or infinity")
            if name not in self.data:
                self.data[name] = np.zeros(
                    (
                        self.episode_length,
                        self.buffer_size,
                        self.num_agents,
                        *array.shape[3:],
                    ),
                    dtype=array.dtype,
                )
            storage = self.data[name]
            if storage.shape[3:] != array.shape[3:] or storage.dtype != array.dtype:
                raise ValueError(f"physics field {name!r} changed shape or dtype")
            storage[:, indices] = array

    def sample_chunks(
        self,
        flat_starts: Sequence[int],
        data_chunk_length: int,
    ) -> dict[str, np.ndarray] | None:
        """PPO의 episode-major flat index와 동일한 연속 sequence를 반환한다."""

        if not self.data:
            return None
        if data_chunk_length <= 0:
            raise ValueError("data_chunk_length must be positive")
        chunks: dict[str, np.ndarray] = {}
        for name, storage in self.data.items():
            axes = (1, 0, *range(2, storage.ndim))
            episode_major = storage.transpose(axes).reshape(
                self.buffer_size * self.episode_length,
                self.num_agents,
                *storage.shape[3:],
            )
            selected = []
            for start in flat_starts:
                if start < 0 or start + data_chunk_length > len(episode_major):
                    raise IndexError("physics sample chunk is outside the buffer")
                # PPO chunk가 episode 경계를 넘으면 RNN state와 physics time pair가
                # 서로 다른 episode를 잇게 되므로 명시적으로 거부한다.
                episode_start = start // self.episode_length
                episode_end = (start + data_chunk_length - 1) // self.episode_length
                if episode_start != episode_end:
                    raise ValueError("physics sample chunk crosses an episode boundary")
                selected.append(episode_major[start : start + data_chunk_length])
            chunks[name] = np.stack(selected, axis=0)
        return chunks

    def clear(self) -> None:
        """할당은 재사용하고 이전 rollout 값만 0으로 지운다."""

        for values in self.data.values():
            values.fill(0)
