"""논문에 기술된 세 과제의 물리적 의미를 한곳에 모은다.

기존 코드는 행동 차원, 위상 이름, 전이 관계를 여러 파일에 숫자로 직접
적어 두었다. 그러면 관측 차원을 바꿀 때 정책, 환경, 손실 함수가 서로 다른
가정을 사용하기 쉽다. 이 모듈의 사양 객체는 앞으로 각 계층이 공유해야 할
단일 기준점이다.

중요한 해석 원칙
-----------------
논문 보충자료의 수식 S126--S155는 가능한 전체 전이 그래프를 정의하지만,
표 S5는 실제 신경망이 출력하는 채널을 더 작게 기술한다. 즉 일부 전이는
학습률 ``lambda``가 아니라 센서 사건/기하 임계값으로 작동한다. 아래에서는
두 종류를 ``learned_transitions``와 ``event_transitions``로 분리해 이 차이를
명시한다. 이것은 논문의 모순을 임의로 감추지 않으면서 실행 가능한 해석을
제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioSpec:
    """한 과제에서 사용하는 위상, 장, 전이와 신경망 입출력 사양.

    Attributes:
        key: CLI와 파일 경로에서 쓰는 짧은 영문 식별자.
        controller_directory: 기존 Webots 컨트롤러 디렉터리 이름.
        title: 문서와 그래프에 표시할 논문상의 과제 이름.
        phases: Macro-ADR 밀도 벡터의 순서. 반응 행렬도 이 순서를 따른다.
        advection_fields: NPC가 조합하는 잠재장 기저의 의미상 이름.
        learned_transitions: 신경망이 전이율을 직접 출력하는 간선.
        event_transitions: 센서 사건이나 기하 조건이 결정하는 간선.
        paper_action_dim: 보충자료 표 S5가 제시한 연속 출력 차원.
        paper_observation_dim: S2.4.2 본문이 최종 ``D_obs``라고 직접 선언한 값.
        paper_table_observation_dim: S162 shared 항과 표 S6의 ``Dim.`` 숫자를
            task-specific 차원으로 해석해 합친 값.
        paper_enumerated_observation_dim: S162와 표 S6에 실제로 이름이 적힌
            feature만 세어 얻은 전체 차원. 원문의 숫자/열거 불일치를 드러낸다.
        legacy_action_dim: 현재 체크포인트가 기대하는 행동 차원.
        legacy_observation_dim: 현재 체크포인트가 기대하는 관측 차원.

    ``legacy_*`` 값은 논문 사양과 현재 구현의 차이를 자동 검사하고, 기존
    체크포인트를 마이그레이션할 때 의도치 않은 호환성 파괴를 막기 위해 둔다.
    """

    key: str
    controller_directory: str
    title: str
    phases: tuple[str, ...]
    advection_fields: tuple[str, ...]
    learned_transitions: tuple[tuple[str, str], ...]
    event_transitions: tuple[tuple[str, str], ...]
    paper_action_dim: int
    paper_observation_dim: int
    paper_table_observation_dim: int
    paper_enumerated_observation_dim: int
    legacy_action_dim: int
    legacy_observation_dim: int

    @property
    def transition_mask(self) -> tuple[tuple[int, ...], ...]:
        """모든 허용 전이를 source-row/target-column 마스크로 반환한다."""

        phase_index = {name: index for index, name in enumerate(self.phases)}
        mask = [[0 for _ in self.phases] for _ in self.phases]
        for source, target in self.learned_transitions + self.event_transitions:
            mask[phase_index[source]][phase_index[target]] = 1
        return tuple(tuple(row) for row in mask)

    @property
    def has_legacy_shape_mismatch(self) -> bool:
        """논문 표와 현재 체크포인트의 입출력 차원이 다른지 알려 준다."""

        return (
            self.paper_action_dim != self.legacy_action_dim
            or self.paper_observation_dim != self.legacy_observation_dim
        )


@dataclass(frozen=True)
class RuntimeMacroSpec:
    """현재 Webots 과제를 실제 궤적 Macro-ADR loss에 연결하는 계약.

    legacy와 paper phase 계약을 별도 key로 보존한다.
    ``learned_reaction_edges``는 ``ParameterLayout.reaction_names``와 같은 순서의
    대표 edge다. 같은 pick rate가 approach와 trail 양쪽의 pickup에 쓰이는 경우는
    ``additional_learned_reaction_edges``의 ``(channel, source, destination)``으로
    명시한다. ``event_reaction_edges=None``인 legacy 계약만 나머지 모든 관측
    phase 변화를 허용하고, paper 계약은 논문의 transition mask를 정확히 열거한다.

    ``arena_size``는 각 병렬 arena의 로컬 ``(width, height)``이고
    ``grid_shape``은 ``(height points, width points)``다. 약 0.1 m 간격은 현재
    E-puck 수와 0.1 m 기본 KDE 폭에서 해상도와 학습 비용의 균형을 이룬다.
    """

    phase_names: tuple[str, ...]
    arena_size: tuple[float, float]
    grid_shape: tuple[int, int]
    learned_reaction_edges: tuple[tuple[int, int], ...]
    additional_learned_reaction_edges: tuple[tuple[int, int, int], ...] = ()
    event_reaction_edges: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        phase_count = len(self.phase_names)
        if phase_count < 2:
            raise ValueError("a macro runtime requires at least two phases")
        if len(set(self.phase_names)) != phase_count:
            raise ValueError("runtime phase names must be unique")
        if min(self.arena_size) <= 0.0:
            raise ValueError("arena dimensions must be positive")
        if min(self.grid_shape) < 3:
            raise ValueError("macro grid must contain at least three points per axis")
        for source, destination in self.learned_reaction_edges:
            if source == destination or not (
                0 <= source < phase_count and 0 <= destination < phase_count
            ):
                raise ValueError("learned reaction edge has invalid phase indices")
        for channel, source, destination in self.additional_learned_reaction_edges:
            if not 0 <= channel < len(self.learned_reaction_edges):
                raise ValueError("additional learned edge has an invalid channel")
            if source == destination or not (
                0 <= source < phase_count and 0 <= destination < phase_count
            ):
                raise ValueError("additional learned edge has invalid phase indices")
        if self.event_reaction_edges is not None:
            for source, destination in self.event_reaction_edges:
                if source == destination or not (
                    0 <= source < phase_count and 0 <= destination < phase_count
                ):
                    raise ValueError("event reaction edge has invalid phase indices")
        all_edges = self.learned_edges + self.resolved_event_reaction_edges
        if len(all_edges) != len(set(all_edges)):
            raise ValueError("runtime reaction edges must be unique")

    @property
    def learned_edges(self) -> tuple[tuple[int, int], ...]:
        """actor channel에 연결된 모든 edge를 실제 residual 순서로 반환한다."""

        additional = tuple(
            (source, destination)
            for _channel, source, destination in self.additional_learned_reaction_edges
        )
        return self.learned_reaction_edges + additional

    @property
    def resolved_event_reaction_edges(self) -> tuple[tuple[int, int], ...]:
        """명시적 paper graph 또는 legacy의 관측 가능한 나머지 edge다."""

        if self.event_reaction_edges is not None:
            return self.event_reaction_edges
        learned = set(self.learned_edges)
        return tuple(
            (source, destination)
            for source in range(len(self.phase_names))
            for destination in range(len(self.phase_names))
            if source != destination and (source, destination) not in learned
        )

    @property
    def all_reaction_edges(self) -> tuple[tuple[int, int], ...]:
        """학습 간선을 먼저 두고 나머지 가능한 event 간선을 뒤에 둔다."""

        return self.learned_edges + self.resolved_event_reaction_edges


PAPER_SCENARIOS: dict[str, ScenarioSpec] = {
    "foraging": ScenarioSpec(
        key="foraging",
        controller_directory="Swarm_Foraging",
        title="Trail-Guided Swarm Foraging",
        phases=("explore", "approach", "home", "trail"),
        advection_fields=("food", "nest", "exploration", "information"),
        # 표 S5의 마지막 두 출력은 pick/drop gate이므로 각각 approach->home,
        # home->explore에 대응한다. resource/trail detection은 관측 event다.
        learned_transitions=(("approach", "home"), ("home", "explore")),
        event_transitions=(
            ("explore", "approach"),
            ("explore", "trail"),
            ("trail", "home"),
        ),
        paper_action_dim=7,
        # shared: velocity 2 + rho/grad 3 + four phases 4 = 9.
        # S2.4.2 declares total D_obs=17, while Table S6 reports 10 task
        # dimensions but names only 9: the three readings are 17/19/18.
        paper_observation_dim=17,
        paper_table_observation_dim=19,
        paper_enumerated_observation_dim=18,
        legacy_action_dim=7,
        legacy_observation_dim=17,
    ),
    "navigation": ScenarioSpec(
        key="navigation",
        controller_directory="Swarm_Navigation",
        title="Formation-Reconfigurable Swarm Navigation",
        phases=("keep", "navigate", "morph", "recover"),
        advection_fields=("flow", "shape"),
        # 표 S5에는 lambda 출력이 없다. S138의 기하 게이트를 사건 전이로 본다.
        learned_transitions=(),
        event_transitions=(
            ("keep", "navigate"),
            ("navigate", "morph"),
            ("morph", "recover"),
            ("recover", "keep"),
        ),
        paper_action_dim=4,
        # S2.4.2 declares 16; shared+Table dimension is 18 and the explicitly
        # enumerated features give 17.
        paper_observation_dim=16,
        paper_table_observation_dim=18,
        paper_enumerated_observation_dim=17,
        legacy_action_dim=4,
        legacy_observation_dim=16,
    ),
    "rescue": ScenarioSpec(
        key="rescue",
        controller_directory="Swarm_Rescue",
        title="Role-Adaptive Swarm Search and Rescue",
        phases=("search", "respond", "relay"),
        advection_fields=("exploration", "target", "relay"),
        # 현재 구현은 respond<->relay 전이율을 학습하는 논문 확장형이다.
        learned_transitions=(("respond", "relay"), ("relay", "respond")),
        event_transitions=(("search", "respond"), ("search", "relay")),
        paper_action_dim=4,
        # S2.4.2 declares 13; shared 8 + six listed/reported task cues gives 14.
        paper_observation_dim=13,
        paper_table_observation_dim=14,
        paper_enumerated_observation_dim=14,
        legacy_action_dim=6,
        legacy_observation_dim=15,
    ),
}


# 체크포인트 호환 Webots 환경의 현재 phase 정의다. 이 registry 때문에 exact
# macro 모드가 논문의 4-phase 설명을 구현한 척하지 않으면서도, 실제 simulator
# trajectory로 ADR residual을 학습할 수 있다. 향후 환경을 4-phase로 이행할 때는
# actor 관측과 함께 이 계약을 바꾸고 기존 profile은 별도 이름으로 보존한다.
RUNTIME_MACRO_SPECS: dict[tuple[str, str], RuntimeMacroSpec] = {
    ("foraging", "legacy"): RuntimeMacroSpec(
        phase_names=("explore", "approach", "carry"),
        arena_size=(2.0, 1.0),
        grid_shape=(11, 21),
        # lambda_pick, lambda_drop
        learned_reaction_edges=((1, 2), (2, 0)),
    ),
    ("foraging", "paper"): RuntimeMacroSpec(
        phase_names=("explore", "approach", "home", "trail"),
        arena_size=(2.0, 1.0),
        grid_shape=(11, 21),
        # Table S5의 pick/drop 두 channel을 app->home, home->exp에 매핑한다.
        # S128에서 trail->home도 같은 pickup gate를 재사용한다.
        learned_reaction_edges=((1, 2), (2, 0)),
        additional_learned_reaction_edges=((0, 3, 2),),
        event_reaction_edges=((0, 1), (0, 3)),
    ),
    ("navigation", "legacy"): RuntimeMacroSpec(
        phase_names=("gather", "transit", "settle"),
        arena_size=(3.0, 1.0),
        grid_shape=(11, 31),
        learned_reaction_edges=(),
    ),
    ("navigation", "paper"): RuntimeMacroSpec(
        phase_names=("keep", "navigate", "morph", "recover"),
        arena_size=(3.0, 1.0),
        grid_shape=(11, 31),
        # Table S5 has no lambda output, so all S138 switches are event-gated.
        learned_reaction_edges=(),
        event_reaction_edges=((0, 1), (1, 2), (2, 3), (3, 0)),
    ),
    ("rescue", "legacy"): RuntimeMacroSpec(
        phase_names=("search", "respond", "relay"),
        arena_size=(3.0, 1.0),
        grid_shape=(11, 31),
        # lambda_anchor, lambda_release
        learned_reaction_edges=((1, 2), (2, 1)),
        event_reaction_edges=((0, 1), (0, 2)),
    ),
    # Rescue already exposes the paper's three roles; only its observation/action
    # table differs, so the phase topology is shared by both contracts.
    ("rescue", "paper"): RuntimeMacroSpec(
        phase_names=("search", "respond", "relay"),
        arena_size=(3.0, 1.0),
        grid_shape=(11, 31),
        learned_reaction_edges=((1, 2), (2, 1)),
        event_reaction_edges=((0, 1), (0, 2)),
    ),
}


def advance_foraging_paper_phases(
    phases: np.ndarray,
    *,
    carrying: np.ndarray,
    resource_detected: np.ndarray,
    trail_available: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """S126--S129의 Foraging graph를 한 control interval 전진시킨다.

    입력 cue를 매번 독립 분류하면 approach에서 trail로 직접 뛰는 등 논문 mask에
    없는 phase 변화가 생긴다. 이 함수는 이전 phase를 명시적으로 받아 한 번에
    허용된 edge 하나만 적용한다. explore에서는 resource detection을 trail보다
    우선하며, pickup/carrying은 approach 또는 trail을 home으로, drop-off는 home을
    explore로 보낸다.
    """

    current = np.asarray(phases, dtype=np.int64)
    shape = current.shape
    carrying_mask = np.asarray(carrying, dtype=np.bool_)
    resource_mask = np.asarray(resource_detected, dtype=np.bool_)
    trail_mask = np.asarray(trail_available, dtype=np.bool_)
    active = (
        np.ones(shape, dtype=np.bool_)
        if active_mask is None
        else np.asarray(active_mask, dtype=np.bool_)
    )
    if any(value.shape != shape for value in (
        carrying_mask,
        resource_mask,
        trail_mask,
        active,
    )):
        raise ValueError("foraging phase inputs must have identical shapes")
    if np.any(((current < 0) | (current >= 4)) & active):
        raise ValueError("active foraging phase must lie in [0,4)")

    next_phase = current.copy()
    exploring = (current == 0) & active
    next_phase[exploring & resource_mask] = 1
    next_phase[exploring & ~resource_mask & trail_mask] = 3
    next_phase[((current == 1) | (current == 3)) & carrying_mask & active] = 2
    next_phase[(current == 2) & ~carrying_mask & active] = 0
    next_phase[~active] = 0
    return next_phase


_ALIASES = {
    "swarm_foraging": "foraging",
    "swarm_navigation": "navigation",
    "swarm_rescue": "rescue",
    "sar": "rescue",
}


def get_scenario(name: str) -> ScenarioSpec:
    """사용자 입력이나 디렉터리 이름을 표준 시나리오 사양으로 변환한다.

    알 수 없는 이름을 조용히 추측하지 않는다. 잘못된 과제 사양으로 학습하면
    체크포인트는 생성되어도 물리 파라미터의 의미가 뒤섞이기 때문이다.
    """

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(normalized, normalized)
    try:
        return PAPER_SCENARIOS[key]
    except KeyError as error:
        choices = ", ".join(sorted(PAPER_SCENARIOS))
        raise ValueError(f"Unknown scenario {name!r}; choose one of: {choices}") from error


def get_runtime_macro_spec(
    name: str,
    *,
    phase_contract: str = "legacy",
) -> RuntimeMacroSpec:
    """scenario alias를 현재 Webots exact-macro 계약으로 해석한다."""

    if phase_contract not in {"legacy", "paper"}:
        raise ValueError("phase_contract must be 'legacy' or 'paper'")
    return RUNTIME_MACRO_SPECS[(get_scenario(name).key, phase_contract)]
