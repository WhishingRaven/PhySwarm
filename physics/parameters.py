"""NPC의 무제한 출력을 논문상의 물리 매니폴드로 사상한다.

논문 식 (22), 보충 식 (S157)--(S159)의 핵심은 신경망이 곧바로 바퀴
속도를 내지 않는다는 점이다. 출력은 먼저 다음 제약을 만족해야 한다.

* advection 가중치 ``omega``: 음수가 아니며 합이 1인 simplex
* 확산 계수 ``D``: ``[D_min, D_max]`` 안의 양수
* 반응률 ``lambda``: 허용 전이에 대해서만 ``[0, lambda_max]``
* 과제별 추가 파라미터: 명시된 닫힌 구간 안의 값

이 모듈은 NumPy 실행 경로를 제공한다. 학습 중 gradient를 보존하는 PyTorch
버전은 ``learning.physics_loss``에 같은 의미로 구현되어 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ParameterLayout:
    """행동 벡터에서 각 물리 파라미터가 차지하는 순서와 변환을 정의한다.

    ``bounded_transform='sigmoid'``는 논문 식 S158의 D/lambda 기준 구현이고,
    advection weight는 두 계약 모두 별도의 softmax/simplex 사상을 쓴다. 체크인된
    모델은 과거 코드의 ``clip[-1,1] -> [0,1]`` 사상으로 학습되었으므로
    ``'clipped_linear'``를 별도의 legacy profile에서만 사용한다. 두 계약을
    이름 없이 섞지 않는 것이 checkpoint 재현성과 새 논문 실험을 동시에
    보존하는 방법이다.
    """

    advection_names: tuple[str, ...]
    reaction_names: tuple[str, ...] = ()
    extra_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    diffusion_bounds: tuple[float, float] = (1.0e-4, 0.035)
    reaction_max: float = 1.0
    advection_scale: float = 0.13
    advection_temperature: float = 1.0
    bounded_transform: Literal["sigmoid", "clipped_linear"] = "sigmoid"
    sparsity_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not self.advection_names:
            raise ValueError("At least one advection field is required")
        if len(set(self.advection_names)) != len(self.advection_names):
            raise ValueError("Advection field names must be unique")
        if len(set(self.reaction_names)) != len(self.reaction_names):
            raise ValueError("Reaction names must be unique")
        d_min, d_max = self.diffusion_bounds
        if not (0.0 <= d_min < d_max):
            raise ValueError("Diffusion bounds must satisfy 0 <= D_min < D_max")
        if self.reaction_max <= 0.0:
            raise ValueError("reaction_max must be positive")
        if self.advection_scale <= 0.0:
            raise ValueError("advection_scale must be positive")
        if self.advection_temperature <= 0.0:
            raise ValueError("advection_temperature must be positive")
        if self.bounded_transform not in {"sigmoid", "clipped_linear"}:
            raise ValueError(
                "bounded_transform must be 'sigmoid' or 'clipped_linear'"
            )
        if not 0.0 <= self.sparsity_threshold < 1.0:
            raise ValueError("sparsity_threshold must satisfy 0 <= threshold < 1")
        for name, (lower, upper) in self.extra_bounds.items():
            if lower >= upper:
                raise ValueError(f"Invalid bounds for {name!r}: {(lower, upper)}")

    @property
    def action_dim(self) -> int:
        """``[omega logits, D, lambda logits, extras]``의 총 차원."""

        return (
            len(self.advection_names)
            + 1
            + len(self.reaction_names)
            + len(self.extra_bounds)
        )


@dataclass(frozen=True)
class ProjectedParameters:
    """물리 제약을 통과한 NPC 출력 묶음.

    ``omega``는 해석 가능한 정규화 비율이고 ``advection_gains``는 실제 속도
    합성에 쓰도록 ``advection_scale``을 곱한 값이다. 두 값을 분리하면 논문의
    simplex 조건과 E-puck의 속도 단위를 동시에 명확히 유지할 수 있다.
    """

    omega: np.ndarray
    advection_gains: np.ndarray
    diffusion: np.ndarray
    reaction_rates: np.ndarray
    extras: dict[str, np.ndarray]

    def as_named_dict(self, layout: ParameterLayout) -> dict[str, np.ndarray]:
        """기존 컨트롤러가 사용하는 ``w_*``/``lambda_*`` 딕셔너리로 변환한다."""

        values: dict[str, np.ndarray] = {
            f"w_{name}": self.advection_gains[..., index]
            for index, name in enumerate(layout.advection_names)
        }
        values["k_diff"] = self.diffusion[..., 0]
        values.update(
            {
                f"lambda_{name}": self.reaction_rates[..., index]
                for index, name in enumerate(layout.reaction_names)
            }
        )
        values.update({name: value[..., 0] for name, value in self.extras.items()})
        # 분석과 검증에서는 정규화 전/후 의미를 구분할 수 있어야 한다.
        values["advection_weights"] = self.omega
        return values


def _stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    """큰 logit에서도 overflow가 발생하지 않는 마지막 축 softmax."""

    scaled = values * temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """극단적인 입력에서도 경고 없이 계산되는 안정적인 sigmoid."""

    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _unit_interval(values: np.ndarray, layout: ParameterLayout) -> np.ndarray:
    """선택한 profile에 따라 latent 값을 ``[0,1]``로 사상한다."""

    if layout.bounded_transform == "sigmoid":
        return _sigmoid(values)
    return (np.clip(values, -1.0, 1.0) + 1.0) / 2.0


def _sparsify_simplex(weights: np.ndarray, threshold: float) -> np.ndarray:
    """legacy Foraging의 작은 weight 억제를 simplex를 보존하며 재현한다."""

    if threshold == 0.0:
        return weights
    suppressed = np.maximum(weights - threshold, 0.0)
    normalizer = np.sum(suppressed, axis=-1, keepdims=True)
    normalized = suppressed / np.maximum(normalizer, 1.0e-8)
    return np.where(normalizer > 1.0e-8, normalized, weights)


def project_numpy_parameters(
    raw_actions: np.ndarray,
    layout: ParameterLayout,
) -> ProjectedParameters:
    """마지막 축의 NPC 출력을 물리적으로 허용된 파라미터로 변환한다.

    Args:
        raw_actions: 임의의 앞쪽 batch 차원을 가진 ``(..., action_dim)`` 배열.
        layout: 각 채널의 의미와 물리 범위를 정의한 사양.

    Returns:
        원래 batch 모양을 보존하는 :class:`ProjectedParameters`.

    Raises:
        ValueError: 행동 차원이 사양과 다르거나 NaN/Inf가 포함된 경우.

    잘못된 값을 자동으로 0으로 바꾸지 않는다. 물리 파라미터 투영 직전의
    비정상 값은 학습 폭주의 중요한 진단 신호이므로 즉시 실패하는 편이 낫다.
    """

    raw = np.asarray(raw_actions, dtype=np.float64)
    if raw.ndim < 1 or raw.shape[-1] != layout.action_dim:
        actual = raw.shape[-1] if raw.ndim else 0
        raise ValueError(
            f"Expected action_dim={layout.action_dim}, received {actual}"
        )
    if not np.isfinite(raw).all():
        raise ValueError("raw_actions contains NaN or infinity")

    cursor = 0
    n_advection = len(layout.advection_names)
    omega = _stable_softmax(
        raw[..., cursor : cursor + n_advection],
        layout.advection_temperature,
    )
    omega = _sparsify_simplex(omega, layout.sparsity_threshold)
    cursor += n_advection

    # paper profile은 식 S158의 bounded sigmoid(D/lambda), legacy profile은
    # checkpoint와 동일한 clipped-linear transform을 사용한다.
    d_min, d_max = layout.diffusion_bounds
    diffusion_unit = _unit_interval(raw[..., cursor : cursor + 1], layout)
    diffusion = d_min + (d_max - d_min) * diffusion_unit
    cursor += 1

    n_reaction = len(layout.reaction_names)
    reaction_rates = layout.reaction_max * _unit_interval(
        raw[..., cursor : cursor + n_reaction],
        layout,
    )
    cursor += n_reaction

    extras: dict[str, np.ndarray] = {}
    for name, (lower, upper) in layout.extra_bounds.items():
        unit_value = _unit_interval(raw[..., cursor : cursor + 1], layout)
        extras[name] = lower + (upper - lower) * unit_value
        cursor += 1

    return ProjectedParameters(
        omega=omega.astype(np.float32),
        advection_gains=(omega * layout.advection_scale).astype(np.float32),
        diffusion=diffusion.astype(np.float32),
        reaction_rates=reaction_rates.astype(np.float32),
        extras={name: value.astype(np.float32) for name, value in extras.items()},
    )
