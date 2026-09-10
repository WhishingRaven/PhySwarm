"""논문 계약과 체크포인트 호환 계약의 action layout registry.

`PAPER_PARAMETER_LAYOUTS`는 보충 식 S158의 softmax/simplex 및 bounded sigmoid
projection과 표 S5의 채널 순서를 따른다. `LEGACY_PARAMETER_LAYOUTS`는 기존
Webots controller/checkpoint가 사용해 온 채널 순서와 clamp-linear transform을
정확히 이름 붙인다. 새 실험과 기존 모델 평가가 같은 무명 상수를 공유하지 않도록
두 registry를 분리한다.
"""

from __future__ import annotations

from typing import Literal

from physics.parameters import ParameterLayout
from tasks.specs import get_scenario


PAPER_PARAMETER_LAYOUTS: dict[str, ParameterLayout] = {
    "foraging": ParameterLayout(
        advection_names=("food", "nest", "rand", "info"),
        reaction_names=("pick", "drop"),
        diffusion_bounds=(1.0e-4, 0.035),
        advection_scale=0.13,
        advection_temperature=1.0,
        bounded_transform="sigmoid",
    ),
    "navigation": ParameterLayout(
        advection_names=("flow", "shape"),
        extra_bounds={"beta": (1.0, 5.0)},
        diffusion_bounds=(1.0e-4, 0.035),
        advection_scale=0.13,
        advection_temperature=1.0,
        bounded_transform="sigmoid",
    ),
    # 표 S5의 순서: exploration, target, relay, diffusion.
    "rescue": ParameterLayout(
        advection_names=("rand", "target", "center"),
        diffusion_bounds=(1.0e-4, 0.035),
        advection_scale=0.13,
        advection_temperature=1.0,
        bounded_transform="sigmoid",
    ),
}


LEGACY_PARAMETER_LAYOUTS: dict[str, ParameterLayout] = {
    "foraging": ParameterLayout(
        advection_names=("food", "nest", "rand", "info"),
        reaction_names=("pick", "drop"),
        diffusion_bounds=(0.0, 0.085),
        advection_scale=0.13,
        advection_temperature=5.0,
        bounded_transform="clipped_linear",
        sparsity_threshold=0.05,
    ),
    "navigation": ParameterLayout(
        advection_names=("flow", "shape"),
        extra_bounds={"beta": (1.0, 5.0)},
        diffusion_bounds=(0.0, 0.035),
        advection_scale=0.13,
        advection_temperature=1.0,
        bounded_transform="clipped_linear",
    ),
    # legacy actor/runtime 순서: target, relay-center, exploration, D,
    # anchor, release. 이는 논문 표의 4채널 Rescue head와 구별한다.
    "rescue": ParameterLayout(
        advection_names=("target", "center", "rand"),
        reaction_names=("anchor", "release"),
        diffusion_bounds=(0.0, 0.035),
        advection_scale=0.13,
        advection_temperature=5.0,
        bounded_transform="clipped_linear",
    ),
}


def get_parameter_layout(
    scenario: str,
    contract: Literal["paper", "legacy"] = "paper",
) -> ParameterLayout:
    """alias를 허용하는 scenario 이름으로 명시적 projection 계약을 반환한다."""

    key = get_scenario(scenario).key
    registry = PAPER_PARAMETER_LAYOUTS if contract == "paper" else LEGACY_PARAMETER_LAYOUTS
    return registry[key]
