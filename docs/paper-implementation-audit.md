# 논문 구현 감사

## 1. 범위와 판정 기준

감사 대상은 저장소 루트의
[Physics-Informed Modeling and Control of Emergent Behaviors in Robot Swarms.pdf](../Physics-Informed%20Modeling%20and%20Control%20of%20Emergent%20Behaviors%20in%20Robot%20Swarms.pdf)
(arXiv:2606.01597v1, 2026-06-01)와 현재 코드다. 논문의 아이디어와 비슷한 계산이
있다는 사실만으로 “구현”으로 판정하지 않았다. 논문이 요구하는 상태, 수식,
학습 데이터, 실행 경로가 모두 연결되어야 **연결됨**이다.

이 문서는 코드 변경 후에도 결론을 추적할 수 있도록 네 상태를 쓴다.

- **연결됨**: runtime/trainer가 실제 사용하고 회귀 테스트가 있음
- **부분 구현**: 목적은 비슷하지만 정의·입력·식이 논문과 다름
- **독립 구현**: 정확한 공통 함수와 테스트가 있으나 공용 trainer에 미연결
- **미구현**: 필요한 코드 또는 데이터 수집 경로가 없음

## 2. 논문의 핵심 계산

phase `σ`의 macro dynamics는 다음 ADR 식이다.

```text
∂ρσ/∂t = -∇·(uσ ρσ) + ∇·(Dσ ∇ρσ) + Rσ(ρ)
```

개체 속도는 macro field를 실행 가능한 motion으로 내리는 Micro-EDM이다.

```text
v = v_adv - D/(ρ + ε) ∇ρ
v_adv = Σk ωk bk,     bk = -∇Φk
```

phase 전이는 event 또는 rate `λχ Δt`로 정의되고, recurrent actor는 제한된
`ω`, `D`, 허용된 `λ`를 출력한다. actor는 로컬 관측만 사용하고 critic은 학습 중
전역 상태를 사용할 수 있다. 최종 목적은 RL loss와 micro/macro physics loss의
결합이다.

## 3. 구현 대응표

| 논문 요구 | 기존 실행 코드의 근거 | 새 공통 구현 | 판정 |
|---|---|---|---|
| 세 과제와 recurrent decentralized actor | `adapters/webots/scenarios`, `learning/rmappo/networks/recurrent.py`, `learning/rmappo/action_heads` | 과제 metadata를 `tasks/specs.py`에 명시 | **연결됨**: actor는 aggregate된 local cue만 받고 GRU를 거친다. 사용되지 않는 `obs_neighbor_num` placeholder는 논문이 요구하지 않는 별도 message-passing 경로다 |
| centralized critic | 각 `rMAPPOPolicy.py`, `critic_rnn.py`, shared observation | 기존 경로 유지 | **연결됨** |
| 제한된 `ω`, `D`, `λ` | 과거에는 환경과 trainer가 서로 다른 상수/후처리를 가짐 | `physics/parameters.py`, `physics/layouts.py`, NumPy/PyTorch 동등 구현 | **연결됨(variant)**: legacy profile은 세 과제에서 공유되고 Foraging/Navigation은 `--parameter_contract paper`로 S158 softmax/simplex + bounded sigmoid profile을 선택 가능. Rescue의 논문 4-action 모호성은 남음 |
| Micro-EDM 속도 | 각 환경의 action-to-velocity mapping과 실행 속도 측정 | `physics/micro_edm.py` | **연결됨**: projector, field basis, density gradient, 실행 속도의 같은-step 정렬을 단위 및 Webots smoke test로 검증 |
| `L_micro`: 실행 속도 대 Micro-EDM | historical surrogate는 “현재 계수로 만든 desired velocity”와 least-squares 이상 계수의 desired velocity를 비교 | `exact_micro_edm_loss`, 별도 side-channel, `--physics_loss_mode exact_micro` | **연결됨(선택형)**: 세 Webots 과제에서 실제 실행 속도와 당시 stochastic/task field basis가 PPO sequence로 전달됨 |
| phase별 KDE `ρσ`, `∇ρσ`, `Δρσ` | 환경은 로컬 밀도/gradient 일부를 관측에 넣음 | NumPy `physics/density.py`, differentiable trajectory KDE `learning/trajectory_loss.py` | **연결됨**: position/phase/active-mask의 명시적 `t→t+1` payload가 PPO sequence를 통과하고 고정 grid KDE를 구성 |
| 보존형 reaction source | transition을 환경별 조건문으로 갱신 | `physics/adr.py`의 column-sum-zero matrix, runtime reaction gate | **연결됨**: 학습 rate는 `λχ`, 비학습 event edge는 관측 phase flux로 구성하며 총 phase 질량을 보존 |
| `L_macro`: 시공간 ADR residual | historical surrogate는 observation의 국소 값과 이상 계수 차이로 residual 구성 | `exact_macro_adr_loss`가 KDE, `∂ρ/∂t`, parameter interpolation, `div(uρ)`, `div(D∇ρ)`, 보존 reaction을 end-to-end 조립 | **연결됨(선택형)**: `--physics_loss_mode exact_macro` 또는 `exact_joint`; body basis를 arena frame으로 회전해 grid에 보간 |
| `L_PINN = L_micro + β L_macro` | 세 trainer가 physics 항을 PPO actor loss에 `pinn_coef`로 추가 | `exact_physics_loss_from_context`, `combine_physics_losses`, soft cap | **연결됨(선택형)**: `exact_joint`와 `--macro_pinn_beta`; raw component와 residual RMS를 별도 기록 |
| `E_ADR`와 Boltzmann reference | 정확한 공통 계산 없음 | `boltzmann_reference_density`, `adr_energy` | **독립 구현** |
| S174--S185 실행/과제 지표 | reward component와 여러 일회성 plot script에 분산 | `experiments/metrics.py`, `analysis/report.py` | **연결됨(offline)**: smoothness, collision, throughput/efficiency/economy, formation error/navigation time, relay connectivity/success/TTC를 테스트; 목표 correspondence 없는 legacy CSV의 formation error는 추측하지 않음 |
| trajectory 증거 보존 | 세 supervisor에서 저장이 꺼져 있고 파일명이 hard-coded되어 분석기와 단절 | 공통 `TrajectoryLogPlan`, `--save_trajectory`, `--trajectory_output` | **연결됨**: 단일-env episode CSV를 충돌 없이 저장하고 공통 분석기로 직접 읽는 실제 Webots 경로를 검증 |
| 동일 설정의 baseline/ablation | `pinn_coef`로 physics 항을 끌 수 있음 | 재현 matrix 및 다중-run 비교기 | **부분 구현**: CI/검정/effect size/paper gap은 자동화됐으나 state-machine/fixed-parameter CLI와 장시간 동일-seed 실험은 아직 없음 |

## 4. 시나리오별 차이

### 4.1 Foraging

- S2.4.2 본문은 최종 `D_obs=17`이라고 명시한다. 그러나 Table S6는 task cue를
  10차원이라고 표기하면서 이름으로 9개만 열거하고, S162 shared cue 9차원과
  합치면 각각 19/18이 된다. legacy 17은 본문 선언값과 같지만 four-phase one-hot을
  포함한 열거 cue와는 맞지 않는다. 세 숫자를 별도 metadata로 보존한다.
- 기본 `--phase_contract legacy`는 search, approach, carry 세 one-hot phase를
  유지한다. `--phase_contract paper`는 explore, approach, home, trail 네 phase를
  실제 observation과 macro KDE에 분리한다. paper mode는 이전 phase를 보존하고
  S129의 다섯 간선만 한 interval에 하나씩 적용한다. resource/trail availability의
  구체적 threshold는 기존 local cue에 기반한 실행 가능한 해석이며 저자 코드로
  확인된 유일한 threshold는 아니다.
- pickup/drop 전이는 Foraging supervisor의 `update_carrying_state`에 구현되어 있다.
- Table S5의 pickup 한 채널은 S128의 approach→home과 trail→home 양쪽에서 같은
  actor tensor/gradient를 공유하고, drop 채널은 home→explore를 제어한다.
- actor head가 과거에는 `[food, nest, random, diffusion, information, pick, drop]`
  순서로 결합되는 반면 환경과 trainer는 `[food, nest, random, information,
  diffusion, pick, drop]`으로 해석했다. 현재 `act_layer.py`의 결합 순서를 후자에
  맞췄고, 각 head bias를 구분해 검증하는 회귀 테스트를 추가했다.
- 환경 diffusion 상한 0.085와 trainer 상한 0.035, runtime에만 있던 sparse
  suppression 불일치를 발견했다. legacy는 `LEGACY_PARAMETER_LAYOUTS`의 0.085
  상한과 동일 suppression을 공유한다. 논문용 bounded-sigmoid profile의 0.035 상한은
  `--parameter_contract paper`로 환경과 trainer에서 함께 선택되고 `config.csv`에
  기록된다.
- 논문 arena는 3 m × 1 m인데 현재 생성 world의 Foraging arena는 2 m × 1 m다.
  이는 방법론 flag로 보정할 수 없는 world-level 재현 차이다.

판정: legacy 호환 경로와 paper-enumerated 18차원/4-phase/S158 projector variant,
exact joint 학습은 모두 실행 가능하다. 다만 누락된 Table S6 cue와 arena 크기 때문에
논문 설정의 완전한 재현으로 판정하지 않는다.

### 4.2 Navigation

- S2.4.2 본문은 최종 `D_obs=16`이라고 명시한다. Table S6의 task-specific 표기는
  9차원이지만 열거 cue는 8개이고 shared cue 9차원과 합치면 18/17이다. legacy
  16, 본문 16, 표 기반 18/17을 같은 값으로 취급하지 않는다.
  `--phase_contract paper`는 keep, navigate, morph, recover 네 one-hot phase와
  열거 가능한 17차원 관측을 사용한다.
- 실행 mapping은 Navigation supervisor의 `_map_actions_to_params`와
  `_calculate_target_wheel_speeds`에서 flow, shape, diffusion, geometry `β`를
  사용한다.
- 일반식은 phase reaction rate 학습을 설명하지만 논문의 Navigation actor 표에는
  `λ` 출력이 없다. 현재 코드는 표를 따르고 phase 전이를 event로 처리한다.
- exact micro는 runtime smoothing 뒤의 실행 속도를 직접 대상으로 삼는다. exact
  macro는 arena-frame field를 사용하고, `β`에 의존하는 shape field는 저장된 centroid
  geometry로 trainer 안에서 다시 계산해 `β` gradient를 보존한다.

판정: paper-enumerated 17차원/4-phase/S158 projector와 exact joint가 연결됐다.
하지만 Table S6의 이름 없는 한 cue와 Navigation reaction-rate 서술/actor 표의
충돌은 원문만으로 해소되지 않는다.

### 4.3 Rescue

- S2.4.2/Table S5는 관측 13차원/actor 4차원을 선언하지만 S162 shared 8과
  Table S6 task cue 6을 합치면 관측은 14다. legacy 구현은 actor 6차원,
  관측 15차원이다.
- 추가 두 action은 anchor/release reaction rate이며, 이는 일반 reaction 모델에는
  자연스럽지만 논문 executable table과는 다르다.
- search에서 respond/relay로 들어가는 일부 전이는 event-deterministic이고,
  Rescue supervisor의 `update_role_and_task_state`에서 anchor/release만 학습된다.
- 과거 `is_robust_test=True` 기본값이 정상 rescued-state 경로를 우회했다. 현재
  표준 과제가 기본(`--no-robust_test`)이고 robustness protocol은 명시적인
  `--robust_test`에서만 켜진다.

판정: legacy 확장과 exact joint는 실행 가능하지만, 14/4 paper 기준 variant는
원문의 reaction graph와 actor 표가 충돌해 아직 제공하지 않는다. 정확 재현으로
볼 수 없으며 두 해석을 분리한 ablation이 필요하다.

## 5. 논문 내부의 명세 충돌

코드만으로 해소할 수 없는 원문 내부 불일치도 있다.

| 항목 | 일반 모델/phase graph | executable actor 표 | 필요한 재현 정책 |
|---|---|---|---|
| Navigation reaction | phase간 `λ`를 학습하는 서술 | actor 4차원에 `λ` 없음 | “표 준수 event 전이”와 “일반식 준수 learned-rate”를 별도 실험 |
| Rescue reaction | search/respond/relay graph에 여러 전이 | actor 4차원은 세 field + D | 논문 표 버전과 legacy 6-action 확장 버전을 구분 |
| Foraging observation | 본문 `D_obs=17`, S162 four-phase shared cue | Table S6 숫자 합 19, 열거 합 18 | 현재 4-phase/18차원은 열거 cue 준수 variant; 세 수치를 metadata에 함께 보존 |
| Navigation observation | 본문 `D_obs=16`, S162 four-phase shared cue | Table S6 숫자 합 18, 열거 합 17 | 현재 4-phase/17차원은 열거 cue 준수 variant; 세 수치를 metadata에 함께 보존 |
| Rescue observation | 본문 `D_obs=13` | S162+Table S6은 14 | paper actor/observation variant를 만들 때 두 계약을 별도 이름으로 분리 |

따라서 하나의 임의 해석을 “정답”으로 고정하지 않는다. `ScenarioSpec`은 paper와
legacy 계약을 함께 기록하고, 실험 ID에 선택한 해석을 포함하도록 한다.

## 6. 현재 검증되는 내용

자동 테스트는 다음 성질을 직접 검증한다.

- NumPy/PyTorch parameter projection의 범위, simplex, gradient
- reaction matrix의 질량 보존
- KDE density의 수치 적분 질량
- differentiable trajectory KDE와 NumPy 기준의 density/gradient/laplacian 일치
- regular-grid flux divergence, `T+1` 상태 정렬, parameter kernel interpolation
- Micro-EDM과 차동구동 변환
- side-channel의 실행 속도 대 Micro-EDM loss 및 actor gradient
- body→arena frame 회전과 명시적 position/phase `T+1` 정렬
- 학습 reaction gate 및 관측 event flux의 보존형 source
- exact joint의 end-to-end micro/macro gradient와 Navigation `β` gradient
- 정상상태 macro residual과 Boltzmann reference/E_ADR
- 강체 이동·회전에 불변인 formation error
- relay graph의 algebraic connectivity
- trajectory/training CSV 분석의 산출물과 원본 불변성
- opt-in 단일-env trajectory 경로, episode 번호, 기존 파일 비덮어쓰기
- legacy 2-row ridge 해의 정규방정식 동등성, gradient, 실제 MPS 실행
- 체크인 모델 load, Foraging action channel 순서, paper/legacy projector·phase 선택
- Rescue 표준 과제 기본값과 명시적 robustness flag

명령은 `conda run -n physwarm pytest -q`다. 현재 78개 test case 중 sandbox에서
76개가 통과하고 MPS hardware test 두 개가 장치 비노출로 skip된다. 두 test 모두 실제
Mac MPS에서 별도로 통과했다. 공용 rMAPPO로 이동하기 전 세 과제
`exact_joint` 4-step Webots smoke가 통과했다. 추가로 Foraging과 Navigation의
paper projector/4-phase 조합도 실제 train update를 통과했고 actor/critic skipped
update가 0이었다. 이 검증은 연결성과 수치 유한성을 증명하지만, 장시간 학습의
논문 수치 재현을 증명하지 않는다. 그것은 다중 seed 실험과 paper table 비교가
별도로 필요하다.

2026-09-10에는 Foraging과 Rescue의 checkpoint-compatible 4-step MPS train update도 실행했다.
각각 actor/critic skip과 parameter repair가 모두 0이었고, 기존 MPS
`linalg.solve` 내부 assert가 재발하지 않았다. 구조 통합 뒤에는 새
`python train.py foraging --webots fast` 경로의 4-step update와 Python
`evaluate.py`/`infer.py`가 체크인 모델을 strict load하는 실제 Webots 실행도 통과했다.

별도의 3-step 실행은 Foraging, Navigation, Rescue에서 각각 8 agents × 3 steps의
trajectory 24행을 저장했다. 공통 분석기는 세 입력 모두에서 중복 agent-step 0,
숫자 결측 0을 확인하고 각 과제별 `summary.json`, `per_step.csv`, task-aware PNG
4개를 생성했다. 모든 과제의 같은-step wheel command가 비영 값이었고 Foraging의
S174 smoothness는 2.889였다. Navigation/Rescue 실패는 각각 명시적 censored flag로
구분됐다. 이는 simulator→CSV→분석 연결을 검증하며 성능 수치의 근거로 사용하지
않는다.

2026-09-09의 4-step smoke에서 기록한 cap 전 원시 loss는 다음과 같다. seed 하나의
수치이므로 성능 비교에는 사용하지 않는다.

| 과제/계약 | `L_macro` | `L_micro` | actor/critic skip |
|---|---:|---:|---:|
| Foraging legacy phase/projector | 3.33467 | 0.06575 | 0 / 0 |
| Navigation legacy phase/projector | 6.85008 | 3.36986 | 0 / 0 |
| Rescue standard task / legacy extension | 0.82611 | 0.01122 | 0 / 0 |
| Foraging paper-enumerated phase/projector | 0.36344 | 0.00573 | 0 / 0 |
| Navigation paper-enumerated phase/projector | 10.33464 | 5.53107 | 0 / 0 |

## 7. 정확 재현까지 남은 필수 작업

1. body/arena frame과 action/state 시간 정렬을 장시간 rollout 및 episode boundary에서
   추가 검증한다. 현재 단위/4-step 검증은 완료됐다.
2. Rescue에 대해 Table S5의 4-action 해석과 기존 6-action learned reaction 해석을
   임의로 섞지 말고 별도 variant/ablation으로 정의한다.
3. S2.4.2의 최종 `D_obs=17/16/13`과 S162+Table S6에서 도출되는
   19/18, 18/17, 14/14 불일치를 저자 자료로 확인한다. 현재 F/N `paper` phase
   계약은 이름이 열거된 cue 합 18/17을 구현한다.
4. Foraging arena를 논문의 3 m × 1 m와 맞춘 별도 world manifest를 만들고 legacy
   2 m × 1 m world와 혼용하지 않는다.
5. Table S9의 Webots 2023b, Ubuntu 20.04, ROS2 Foxy 실행 환경과 현재 검증 환경인
   Webots R2025a, macOS의 차이를 별도 재현 축으로 관리한다.
6. 논문과 동일한 로봇 수, horizon, 장애물/목표 배치, seed 수를 확정해 manifest로
   저장한다.
7. 구현된 평균·bootstrap CI·paper gap·effect size 위에 metric 방향, 실패 episode,
   다중 검정, practical significance threshold를 사전 등록한다.
8. state-machine/fixed-parameter baseline을 명시적 variant로 만든 뒤 multi-seed
   장시간 baseline/legacy/exact micro/exact macro/exact joint ablation으로 논문의
   표와 그림을 수치적으로 비교한다.

현재 exact micro/macro/joint는 세 Webots 실행 경로에 연결됐다. 따라서 “논문식
PINN 손실의 실행 가능한 구현”이라고는 할 수 있지만, 위 명세 충돌과 world·장시간
실험 차이가 남아 있어 “논문 결과 완전 재현”이라고 표현하지 않는다.
