# PhySwarm 프로젝트 구조

이 문서는 현재 저장소의 디렉토리 구조와 파일별 역할을 정리한 문서입니다. 프로젝트는 Webots 기반 군집 로봇 시뮬레이션과 MAPPO(Recurrent Multi-Agent PPO) 학습을 제공하며, `Swarm_Foraging`, `Swarm_Navigation`, `Swarm_Rescue` 세 가지 시나리오를 포함합니다.

## 1. 디렉토리 구조

```text
PhySwarm/
├── README.md
├── MIGRATION.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── .gitignore
├── controllers/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── csv_robot.py
│   │   ├── device.py
│   │   ├── gymnasium_api.py
│   │   ├── webots_env.py
│   │   ├── webots_launcher.sh
│   │   └── webots_runtime.py
│   ├── epuck_controller/
│   │   └── epuck_controller.py
│   ├── Swarm_Foraging/
│   │   ├── config.csv
│   │   ├── progress_eval.csv
│   │   ├── logs/
│   │   ├── models/
│   │   ├── plot/
│   │   └── supervisor_controller/
│   ├── Swarm_Navigation/
│   │   ├── models/
│   │   ├── plot/
│   │   └── supervisor_controller/
│   └── Swarm_Rescue/
│       ├── models/
│       ├── plot/
│       └── supervisor_controller/
├── tests/
│   └── test_macos_migration.py
└── worlds/
    ├── generate_wbt.py
    └── generated_world.wbt
```

각 시나리오의 `supervisor_controller/`는 다음 공통 구조를 사용합니다.

```text
supervisor_controller/
├── base_runner.py
├── config.py
├── prey_runner.py
├── supervisor_rlcontroller.py
├── train_prey.py
├── train_mappo.sh                 # Navigation, Rescue에 존재
├── utilities.py                   # Navigation, Rescue에 존재
├── r_mappo/
│   ├── __init__.py
│   ├── r_mappo.py
│   ├── algorithm/
│   │   ├── __init__.py
│   │   ├── act_layer.py
│   │   ├── critic_rnn.py
│   │   ├── rMAPPOPolicy.py
│   │   ├── rnn.py
│   │   └── v_out.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── mlp_policy.py
│   │   ├── recurrent_policy.py
│   │   └── trainer.py
│   └── utils/                     # Navigation, Rescue에 존재
│       ├── __init__.py
│       ├── dict2namedtuple.py
│       ├── popart.py
│       ├── ppo_buffer.py
│       ├── segment_tree.py
│       ├── util.py
│       └── valuenorm.py
```

## 2. 루트 파일

| 파일 | 역할 |
|---|---|
| `README.md` | 프로젝트 목적, 실행 방법, 시뮬레이션 및 학습 개요를 설명하는 기본 안내서 |
| `MIGRATION.md` | macOS 또는 기존 실행 환경에서 현재 구조로 이전할 때 필요한 변경 사항과 호환성 안내 |
| `LICENSE` | 프로젝트 라이선스 전문 |
| `environment.yml` | Conda 환경 이름과 Python 및 주요 패키지 의존성 정의 |
| `requirements.txt` | pip로 설치할 Python 패키지 목록 |
| `.gitignore` | 캐시, 로컬 환경, 생성 로그 및 기타 버전 관리 제외 항목 정의 |
| `.DS_Store` | macOS Finder가 생성한 로컬 메타데이터 파일. 애플리케이션 로직에는 사용되지 않음 |

## 3. 공통 컨트롤러와 Webots 연동

| 파일 | 역할 |
|---|---|
| `controllers/common/__init__.py` | 공통 컨트롤러 모듈 초기화 |
| `controllers/common/device.py` | Webots 장치(Device)를 찾고 센서·액추에이터 접근을 공통화 |
| `controllers/common/csv_robot.py` | CSV 설정 또는 기록을 사용하는 로봇 동작과 데이터를 처리 |
| `controllers/common/webots_env.py` | Webots 시뮬레이션을 Gymnasium 환경 형태로 감싸 관측, 행동, 보상, 종료를 연결 |
| `controllers/common/gymnasium_api.py` | Gymnasium의 환경 API와 Webots 환경 사이의 호환 계층 제공 |
| `controllers/common/webots_runtime.py` | Webots 프로세스 및 런타임 초기화·종료와 실행 상태 관리 |
| `controllers/common/webots_launcher.sh` | 공통 Webots 런타임을 셸에서 시작하는 실행 스크립트 |
| `controllers/epuck_controller/epuck_controller.py` | e-puck 개별 로봇의 센서 입력과 모터 제어를 담당하는 Webots 로봇 컨트롤러 |

## 4. 시나리오별 파일

### 4.1 시나리오 상위 파일

| 파일 | 역할 |
|---|---|
| `controllers/Swarm_Foraging/config.csv` | Foraging 실험의 로봇 수, 환경 및 학습 관련 설정값 |
| `controllers/Swarm_Foraging/progress_eval.csv` | Foraging 평가 실행의 진행 상황과 평가 결과를 저장하는 CSV |
| `controllers/Swarm_Navigation/` | 목표 위치 이동 또는 탐색을 위한 Navigation 시나리오 영역 |
| `controllers/Swarm_Rescue/` | 구조 작업을 위한 Rescue 시나리오 영역 |

각 시나리오의 다음 파일은 동일한 학습 실행 구조를 사용합니다.

| 파일 | 역할 |
|---|---|
| `controllers/{scenario}/supervisor_controller/config.py` | 해당 시나리오의 환경, 네트워크, PPO 및 실행 하이퍼파라미터 구성 |
| `controllers/{scenario}/supervisor_controller/base_runner.py` | 환경과 에이전트를 연결하고 episode/rollout 실행을 조정하는 기본 실행기 |
| `controllers/{scenario}/supervisor_controller/prey_runner.py` | prey 또는 작업 대상 로봇의 episode 실행과 제어를 담당 |
| `controllers/{scenario}/supervisor_controller/supervisor_rlcontroller.py` | Webots Supervisor에서 RL 정책을 호출하고 군집 상태·보상·행동을 연결하는 진입점 |
| `controllers/{scenario}/supervisor_controller/train_prey.py` | prey 정책 또는 관련 환경을 학습·실행하는 Python 진입 스크립트 |
| `controllers/{scenario}/supervisor_controller/train_mappo.sh` | MAPPO 학습 명령을 실행하는 셸 스크립트. Navigation과 Rescue에 포함 |
| `controllers/{scenario}/supervisor_controller/utilities.py` | Navigation과 Rescue 실행에서 사용하는 보조 함수와 데이터 처리 유틸리티 |

`{scenario}`에 해당하는 실제 경로는 다음 세 가지입니다.

- `controllers/Swarm_Foraging`
- `controllers/Swarm_Navigation`
- `controllers/Swarm_Rescue`

### 4.2 Recurrent MAPPO 구현

아래 파일은 세 시나리오에 각각 존재합니다. 즉, 다음 표의 경로마다 하나의 실제 파일이 있습니다.

| 실제 경로 | 역할 |
|---|---|
| `controllers/{scenario}/supervisor_controller/r_mappo/__init__.py` | MAPPO 패키지 초기화 |
| `controllers/{scenario}/supervisor_controller/r_mappo/r_mappo.py` | recurrent MAPPO 알고리즘의 학습 업데이트, advantage 처리 및 정책 최적화 |
| `controllers/{scenario}/supervisor_controller/r_mappo/algorithm/__init__.py` | 알고리즘 구성요소 패키지 초기화 |
| `controllers/{scenario}/supervisor_controller/r_mappo/algorithm/act_layer.py` | 정책 출력에서 행동 분포를 만들고 행동을 샘플링·평가 |
| `controllers/{scenario}/supervisor_controller/r_mappo/algorithm/critic_rnn.py` | 순환 구조를 사용하는 critic/value 네트워크 |
| `controllers/{scenario}/supervisor_controller/r_mappo/algorithm/rMAPPOPolicy.py` | actor와 critic을 묶은 recurrent MAPPO 정책 인터페이스 |
| `controllers/{scenario}/supervisor_controller/r_mappo/algorithm/rnn.py` | 순환 신경망 기반 actor/feature 처리와 hidden state 관리 |
| `controllers/{scenario}/supervisor_controller/r_mappo/algorithm/v_out.py` | value 네트워크의 출력층 및 value prediction 처리 |
| `controllers/{scenario}/supervisor_controller/r_mappo/base/__init__.py` | 기본 정책·학습기 패키지 초기화 |
| `controllers/{scenario}/supervisor_controller/r_mappo/base/mlp_policy.py` | MLP 기반 정책 네트워크 정의 |
| `controllers/{scenario}/supervisor_controller/r_mappo/base/recurrent_policy.py` | recurrent 정책의 공통 구성과 순전파 인터페이스 |
| `controllers/{scenario}/supervisor_controller/r_mappo/base/trainer.py` | rollout 데이터를 사용한 학습 루프와 optimizer 실행 |

실제 반복 경로는 `{scenario}`를 치환한 다음 3개입니다.

- `controllers/Swarm_Foraging/supervisor_controller/r_mappo/...`
- `controllers/Swarm_Navigation/supervisor_controller/r_mappo/...`
- `controllers/Swarm_Rescue/supervisor_controller/r_mappo/...`

### 4.3 MAPPO 보조 유틸리티

다음 파일은 Navigation과 Rescue의 `supervisor_controller/r_mappo/utils/`에 각각 존재합니다.

| 파일 | 역할 |
|---|---|
| `utils/__init__.py` | 보조 유틸리티 패키지 초기화 |
| `utils/dict2namedtuple.py` | 딕셔너리 설정을 attribute 방식으로 접근 가능한 named tuple 형태로 변환 |
| `utils/popart.py` | value target의 통계 변화에 맞춰 출력층을 보정하는 PopArt 정규화 |
| `utils/ppo_buffer.py` | PPO rollout transition, action log-probability, reward, value를 저장 |
| `utils/segment_tree.py` | 우선순위 또는 구간 질의를 위한 segment tree 자료구조 |
| `utils/util.py` | 학습 시 사용하는 초기화, gradient, tensor 및 기타 공통 보조 함수 |
| `utils/valuenorm.py` | value target의 평균·분산을 추적하는 value normalization |

실제 경로:

- `controllers/Swarm_Navigation/supervisor_controller/r_mappo/utils/`
- `controllers/Swarm_Rescue/supervisor_controller/r_mappo/utils/`

## 5. 시각화와 실험 데이터

### Foraging

`controllers/Swarm_Foraging/plot/`의 스크립트는 학습·평가 결과를 그래프나 영상 프레임으로 변환합니다.

| 파일 | 역할 |
|---|---|
| `draw_ci.py` | 신뢰구간 또는 confidence interval 시각화 |
| `draw_elasticity_n.py` | 로봇 수 변화에 대한 성능 탄력성 시각화 |
| `draw_error.py` | 오차 및 변동성 그래프 생성 |
| `draw_expand_n.py` | 군집 규모 확장 실험 결과 시각화 |
| `draw_field_video_n.py` | field 실험의 프레임·영상 결과 생성 |
| `draw_params.py` | 학습 파라미터별 결과 비교 |
| `draw_pattern.py` | 군집 행동 패턴 시각화 |
| `draw_radar.py` | 여러 평가 지표를 radar chart로 표시 |
| `draw_snap.py` | 시뮬레이션 snapshot 또는 시점별 결과 시각화 |

### Navigation

Navigation의 `plot/`에는 Foraging과 공통인 `draw_ci.py`, `draw_elasticity_n.py`, `draw_error.py`, `draw_expand_n.py`, `draw_field_video_n.py`, `draw_params.py`, `draw_pattern.py`, `draw_radar.py`, `draw_snap.py` 외에 다음 분석 스크립트가 있습니다.

| 파일 | 역할 |
|---|---|
| `adv_diff.py` | advection와 diffusion 관련 결과 비교 |
| `advection.py` | advection 성분 또는 실험 데이터 분석 |
| `diffusion.py` | diffusion 성분 또는 실험 데이터 분석 |
| `draw_dict_adv.py` | advection 조건을 딕셔너리 단위로 비교 |
| `draw_dict_diff.py` | diffusion 조건을 딕셔너리 단위로 비교 |
| `reaction.py` | reaction 성분 또는 반응 조건 분석 |

### Rescue

`controllers/Swarm_Rescue/plot/`에는 `draw_ci.py`, `draw_elasticity_n.py`, `draw_error.py`, `draw_expand_n.py`, `draw_field_video_n.py`, `draw_params.py`, `draw_pattern.py`, `draw_radar.py`, `draw_snap.py`가 있으며, Rescue 평가 결과를 같은 방식으로 시각화합니다.

각 시나리오의 `plot/data/` 디렉토리는 플롯 스크립트가 읽는 실험 입력 데이터와 중간 결과를 보관하는 위치입니다. 데이터 파일이 생성되거나 교체될 수 있으므로 소스 코드와 분리되어 있습니다.

## 6. 모델과 로그 산출물

각 시나리오의 `models/`에는 다음 파일 세트가 있습니다.

| 경로 패턴 | 역할 |
|---|---|
| `controllers/{scenario}/models/checkpoint.pt` | 학습 재개 또는 평가에 사용하는 전체 학습 체크포인트 |
| `controllers/{scenario}/models/policy_0/act.pt` | 정책의 action layer 가중치 |
| `controllers/{scenario}/models/policy_0/rnn_network.pt` | recurrent actor 또는 정책 네트워크 가중치 |
| `controllers/{scenario}/models/policy_0/rnn_critic.pt` | recurrent critic 네트워크 가중치 |
| `controllers/{scenario}/models/policy_0/v_network.pt` | value 출력 네트워크 가중치 |

`{scenario}`는 `Swarm_Foraging`, `Swarm_Navigation`, `Swarm_Rescue` 각각으로 치환됩니다. `.pt` 파일은 PyTorch 바이너리 산출물이므로 소스처럼 직접 읽는 파일이 아니라 학습된 정책을 재사용하기 위한 파일입니다.

Foraging의 `controllers/Swarm_Foraging/logs/`에는 다음이 있습니다.

| 파일 | 역할 |
|---|---|
| `events.out.tfevents.*` | TensorBoard가 기록한 학습·평가 scalar 이벤트 로그. timestamp와 호스트명이 파일명에 포함됨 |
| `summary.json` | 실행 요약 및 로그 메타데이터 |

`events.out.tfevents.*`는 현재 저장소에 여러 실행분이 존재하는 로그 파일 묶음이며, 개별 파일은 동일한 TensorBoard 이벤트 포맷을 사용합니다.

## 7. Webots 월드와 테스트

| 파일 | 역할 |
|---|---|
| `worlds/generate_wbt.py` | 시뮬레이션 로봇, 장애물 및 환경 요소를 조합해 Webots `.wbt` 월드를 생성 |
| `worlds/generated_world.wbt` | 생성된 Webots 월드 정의. 시뮬레이션에서 직접 로드되는 장면 파일 |
| `tests/test_macos_migration.py` | macOS 환경 이전 후 경로, 런타임 또는 실행 호환성을 확인하는 회귀 테스트 |

## 8. 실행 흐름 요약

```text
Webots world
    -> controllers/common/webots_runtime.py
    -> controllers/common/webots_env.py / gymnasium_api.py
    -> supervisor_controller/supervisor_rlcontroller.py
    -> base_runner.py + prey_runner.py
    -> r_mappo/r_mappo.py
    -> models/policy_0/*.pt
    -> logs/ 또는 plot/ 결과
```

일반적인 개발·실행 순서는 다음과 같습니다.

1. `environment.yml` 또는 `requirements.txt`로 Python 환경을 준비합니다.
2. `worlds/generate_wbt.py`로 필요한 Webots 월드를 생성하거나 `generated_world.wbt`를 사용합니다.
3. 공통 Webots 런타임과 시나리오별 Supervisor 컨트롤러를 실행합니다.
4. `train_mappo.sh` 또는 `train_prey.py`로 학습을 수행합니다.
5. 생성된 체크포인트와 로그를 사용해 평가하고, `plot/`의 분석 스크립트로 결과를 시각화합니다.

`__pycache__/`와 `.pytest_cache/`는 Python 실행 및 테스트 과정에서 자동 생성되는 캐시이며, 프로젝트 동작을 정의하는 소스 파일이 아닙니다.