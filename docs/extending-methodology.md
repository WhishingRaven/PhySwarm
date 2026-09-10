# 과제를 보존하면서 방법론 확장하기

## 1. 무엇을 고정하고 무엇을 바꿀 것인가

새 방법론을 실험할 때 먼저 변경 축을 하나로 제한한다.

| 고정 대상 | 이유 |
|---|---|
| Webots world, robot 수, sensor/actuator, horizon | 물리 환경 난이도를 동일하게 유지 |
| 관측 가능 정보와 centralized-training 조건 | 정보 우위로 인한 비교 왜곡 방지 |
| task reward와 success condition | 같은 과제를 푸는지 보장 |
| seed와 초기 배치 목록 | paired 비교 가능 |
| 평가 metric 및 실패 episode 처리 | 유리한 지표 선택 방지 |

교체 가능한 축은 policy backbone, multi-agent optimizer, parameter projector,
Micro-EDM, density estimator, reaction model, physics loss weighting이다. 두 축 이상을
동시에 바꾸면 어떤 변경이 효과를 냈는지 분리할 ablation을 추가한다.

## 2. 새 정책을 붙이는 최소 인터페이스

정책은 다음 개념적 계약을 만족해야 한다.

```python
raw_action, next_hidden = actor(local_observation, previous_action, hidden)
value = critic(shared_observation, critic_hidden)
physical = projector(raw_action, scenario_spec)
```

actor의 raw action 차원과 순서는 `ScenarioSpec` 및 `ParameterLayout`이 결정한다.
환경이 raw index를 직접 해석하지 않고 projector 결과의 이름 있는 필드를 받도록
현재 이행했으며, 기존 checkpoint는 명시적인 legacy layout으로 유지한다. 새
논문 실험은 Foraging/Navigation에서 `--parameter_contract paper`를 선택하고 run의
`config.csv`에 계약을 남긴다. Rescue의 4-action paper layout은 원문의 actor 표와
reaction-rate 서술을 먼저 별도 variant로 정의하기 전에는 추가하지 않는다.

## 3. 새 물리 모델을 붙이는 최소 인터페이스

Micro model은 다음 데이터를 받는다.

- named vector-field basis `(..., K, 2)`
- density `rho`, gradient `grad_rho`
- projected `omega/gain`, `D`, 수치 안정화 `epsilon`

출력은 `advection`, `diffusion`, `total velocity`로 분해해 반환한다. 합계만
반환하면 어느 항이 실패했는지 분석할 수 없다.

Macro model은 phase-first tensor 계약을 권장한다.

```text
rho                  (..., S, H, W)
drho_dt              (..., S, H, W)
flux_divergence      (..., S, H, W)
laplacian_rho        (..., S, H, W)
reaction_matrix      (..., S, S)
```

reaction matrix의 `(destination, source)` convention과 column-sum-zero 불변식을
테스트한다. 다른 PDE나 graph neural operator로 바꾸더라도 동일 shape와 진단
출력을 제공하면 trainer와 분석기를 재사용할 수 있다.

## 4. `PhysicsContext` 설계

정확한 논문 손실을 연결할 때 actor observation을 억지로 늘리지 않는다. 환경의
`info` 또는 명시적 side channel로 다음 immutable payload를 runner에 전달한다.

```text
PhysicsContext
├── position_start/end[transition, env, agent, xy]
├── phase_start/end[transition, env, agent]
├── active_start/end[transition, env, agent]
├── executed_velocity[transition, env, agent, xy]
├── field_bases[transition, env, agent, field, body_xy]
├── field_bases_global[transition, env, agent, field, arena_xy]
├── density, density_gradient, active_mask
├── reaction_gates[transition, env, agent, learned_edge]
└── optional geometry needed to rebuild parameter-dependent fields
```

runner는 context를 rollout buffer에 넣고, sampler는 PPO sequence와 같은 index로
physics batch를 자른다. position/phase endpoint를 transition마다 둘 다 저장하므로
`t,t+1`을 observation에서 추측하지 않는다. 공통 `PhysicsContextBuffer`는 RNN
chunk가 episode 경계를 넘거나 연속한 endpoint가 맞지 않으면 즉시 실패한다.

현재 위 core payload는 세 과제에서 모두 연결됐다. collocation grid는 배열을 매
step 복제하지 않고 `RuntimeMacroSpec`의 arena bounds/grid shape로 trainer에서
재구성한다. Navigation의 shape basis처럼 actor parameter에 의존하는 field는
`auxiliary_fields`의 최소 geometry를 이용해 trainer에서 다시 계산해 gradient를
보존한다. 새 시나리오도 관측 벡터에 physics-only 값을 끼워 넣지 말고 이 계약을
확장해야 한다.

## 5. 손실 교체 절차

1. 새 손실 함수를 `learning/`에 순수 함수로 추가한다.
2. analytic toy field에서 residual이 0이 되는 단위 테스트와 gradient finite
   테스트를 만든다.
3. 고정된 rollout fixture에 대해 loss component를 snapshot한다.
4. config flag로 `legacy`, `exact_micro`, `exact_macro`, `exact_joint`, off
   (`pinn_coef=0`) variant를 동시에 유지한다.
5. actor update에 연결하고 `RL`, `micro`, `macro`, gradient norm을 각각 기록한다.
6. 동일 checkpoint/seed의 짧은 smoke run에서 NaN, 메모리, 처리량을 확인한다.
7. 다중 seed ablation이 끝난 뒤에만 legacy path 제거를 검토한다.

권장 총손실 표기는 다음처럼 각 계수를 명시한다.

```text
L_actor_total = L_PPO - c_entropy H + c_micro L_micro + c_macro L_macro
```

현재 `pinn_coef`는 전체 physics 항, `macro_pinn_beta`는 joint 안의 macro 상대
가중치로 작동하므로 `c_micro=pinn_coef`,
`c_macro=pinn_coef*macro_pinn_beta`다. cap 전 두 원시 loss와 residual RMS를 반드시
함께 기록해 scale 차이가 최종 합에 가려지지 않게 한다.

## 6. 새 시나리오 추가 절차

1. `ScenarioSpec`에 phase, field, transition mask, paper/legacy dimensions를 추가한다.
2. task reward와 success condition을 물리 loss와 분리해 문서화한다.
3. Webots supervisor는 공통 Gymnasium 계약을 구현한다.
4. named parameter projection과 action ordering 테스트를 추가한다.
5. collision, efficiency/error/connectivity 중 적합한 기본 metric과 task-specific
   metric을 정의한다.
6. 최소 trajectory fixture로 공통 분석 보고서를 검증한다.
7. random policy, MAPPO-only, physics variant의 smoke run을 순서대로 실행한다.

## 7. 호환성과 버전 관리

- observation/action layout을 바꾸면 문자열 버전(예: `foraging-paper-v1`,
  `rescue-legacy-v1`)을 checkpoint metadata에 저장한다.
- state dict module 이름을 바꾸면 명시적 key converter와 양방향 fixture를 둔다.
- 과거 CSV 열 이름을 바꾸지 말고 새 canonical 이름으로 변환하는 importer를 둔다.
- world geometry 변경은 method change와 같은 commit/experiment에 섞지 않는다.
- deprecated wrapper는 실제 호출이 없음을 확인한 뒤 별도 정리 변경에서 제거한다.

## 8. 완료 조건

방법론 변경은 다음을 모두 만족할 때 완료다.

- 수식 단위 테스트와 전체 회귀 테스트 통과
- Webots short train/eval에서 finite action, reward, loss 확인
- 기존 baseline checkpoint가 여전히 load되거나 명시적 converter 제공
- 동일 seed에서 baseline과 candidate 산출물 생성
- `summary.json`, raw CSV, config, checkpoint, 환경 manifest 보존
- 논문 감사표와 재현 문서의 상태 갱신

이 조건은 성능 향상을 보장하는 기준이 아니라, 비교 결과를 믿을 수 있게 만드는
최소 공학 기준이다.
