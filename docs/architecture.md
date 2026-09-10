# 아키텍처와 책임 경계

## 설계 원칙

PhySwarm은 디렉터리 하나가 하나의 변경 이유를 갖도록 나눈다. 물리식은
시뮬레이터를 모르고, 학습기는 Webots 객체를 모르며, Webots와 Gymnasium의 API
경계는 각각 별도 adapter가 소유한다. 세 과제는 동일한 학습 수명주기와 rMAPPO
코어를 사용하고 실제로 다른 환경·action head·물리 정규화만 선택한다.

## 현재 구조

```text
PhySwarm/
├── train.py / evaluate.py / infer.py  # 사용자 Python 진입점
├── app/                               # 실행 수명주기와 controller subprocess
├── tasks/
│   ├── specs.py                       # phase/field/action 계약
│   └── profiles.py                    # 과제별 재현 기본값
├── physics/                           # 순수 NumPy/Torch 물리 모델
├── learning/
│   ├── physics_loss.py
│   ├── rollout_context.py
│   ├── trajectory_loss.py
│   └── rmappo/
│       ├── runner.py, rollout.py      # 세 과제 공용 실행기
│       ├── buffer.py, policy.py       # 세 과제 공용 PPO 상태
│       ├── networks/                  # actor/critic/value 공용 network
│       ├── action_heads/              # 과제별 출력 의미와 checkpoint 이름
│       └── trainers/                  # 과제별 물리 정규화 전략
├── adapters/
│   ├── gymnasium/                     # terminated/truncated 의미
│   └── webots/
│       ├── launcher.py, runtime.py     # Webots 프로세스/API 경계
│       ├── gymnasium_env.py            # Webots Supervisor ↔ Gymnasium
│       ├── csv_robot.py                # emitter/receiver transport
│       └── scenarios/                  # 세 Webots 과제 환경
├── experiments/                       # 궤적 계약과 지표
├── analysis/                          # 보고서와 비교 CLI
├── controllers/epuck_controller/      # Webots가 이름으로 시작하는 로봇 process
├── worlds/                            # Webots world
└── tests/
```

기존 `controllers/Swarm_*/supervisor_controller`에 있던 세 벌의 config, runner,
buffer, policy, network와 shell launcher는 제거됐다. 체크인된 model·plot data는
실험 자산이므로 원래 과제 디렉터리에 보존한다.

## 실행과 의존 방향

```text
train.py / evaluate.py / infer.py
    -> app.execute -> adapters.webots.launcher
    -> app.controller_process -> app.training
        -> adapters.webots.scenarios[task]
        -> learning.rmappo.registry[task]
            -> shared runner / buffer / policy / networks
            -> task action head + physics trainer
        -> physics + tasks

trajectory/training CSV
    -> analysis -> experiments.metrics
```

`physics`, `tasks`, `experiments.metrics`는 `controller`를 import하지 않는다.
`learning.rmappo`도 Webots 환경 클래스를 import하지 않는다. Webots의 node,
receiver, emitter, motor와 native API 설정은 `adapters.webots`만 소유한다.

## 과제와 체크포인트 계약

| 과제 | phase | actor 출력 | 체크포인트 관측/행동 |
|---|---|---:|---:|
| Foraging | explore, approach, home, trail | 7 | 17 / 7 |
| Navigation | keep, navigate, morph, recover | 4 | 16 / 4 |
| Rescue | search, respond, relay | 논문 4 / 현재 6 | 15 / 6 |

공용 policy는 과제별 action head를 주입받는다. head의 field 이름은 기존
`state_dict` key를 유지하므로 체크인된 세 과제 모델을 strict load할 수 있다.
NumPy 실행 경로의 `project_numpy_parameters`와 PyTorch 학습 경로의
`project_torch_parameters`는 `physics.layouts`의 동일한 layout을 사용한다.

## rollout 물리 side-channel

정확한 물리 손실은 actor observation을 억지로 재해석하지 않는다. 환경은 각
control interval에 위치, phase, active mask, arena-frame field basis, 실행 속도,
density/gradient와 reaction gate를 별도 payload로 반환한다. 공용 rollout과 buffer가
PPO sequence와 같은 index로 이를 자르고 `learning.trajectory_loss`가 Micro-EDM과
Macro-ADR residual을 계산한다.

## 변경 불변식

- `reset()`과 `step()`은 Gymnasium의 최신 tuple 계약을 유지한다.
- action index, projector layout, action head channel 순서는 일치해야 한다.
- centralized critic state와 decentralized actor observation을 섞지 않는다.
- reaction matrix의 각 열 합은 0이어야 한다.
- 분석기는 입력 CSV를 수정하지 않는다.
- 궤적 파일은 단일 environment만 기록하고 기존 파일을 덮어쓰지 않는다.
- 체크포인트 module 이름을 바꾸면 변환기나 버전 계약을 함께 제공한다.
- Webots 프로세스는 Python context manager가 소유하고 성공·실패·interrupt 모두에서
  종료한다.
