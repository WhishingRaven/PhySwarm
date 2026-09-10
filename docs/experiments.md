# 실험 지표와 보고서

## 1. 목적

기존 `controllers/Swarm_*/plot`은 특정 논문 그림, 파일명, 로컬 데이터 배치에
결합된 스크립트가 많다. 공통 분석기 `analysis`는 입력과 출력 경로를
명시하고 원본 CSV를 수정하지 않으며, 같은 입력에서 같은 JSON/CSV/PNG를 만든다.
기존 plot은 논문 그림 재제작용으로 남기고, 새 실험의 비교 기준은 공통 분석기를
사용한다.

## 2. 실행

```bash
python -m analysis \
  --input path/to/data.csv \
  --output path/to/report \
  --scenario auto
```

`--scenario`는 `auto`, `foraging`, `navigation`, `rescue` 중 하나다. auto는
`w_food/w_nest`, `w_flow/w_shape`, `role/chain_connected`처럼 겹치지 않는 열로
보수적으로 판별한다. 판별되지 않으면 공통 그래프만 만들며 임의로 과제를
추측하지 않는다.

세 Webots launcher에서 입력 CSV를 직접 만들려면
`--save_trajectory --trajectory_output /absolute/path/to/run.csv`를 전달한다.
저장은 한 environment의 episode마다 별도 파일에 exclusive-create 방식으로
수행되므로 원본 실험을 자동 덮어쓰지 않는다. 상세 경로 규칙은
[reproduction.md](reproduction.md#7-실행-산출물)에 있다.

## 3. 입력 스키마

### 3.1 Trajectory CSV

필수 열은 다음 네 개다.

| 열 | 의미 |
|---|---|
| `step` | simulation step |
| `agent_id` | episode 안에서 안정적인 로봇 ID |
| `pos_x`, `pos_y` | 동일 단위·좌표계의 2차원 위치 |

있으면 사용하는 선택 열은 다음과 같다.

- 공통: `is_alive`, `is_colliding`, wheel command pair
  (`cmd_omega_l/r` 또는 `left_speed/right_speed`), `episode_max_steps`
- Foraging: `is_carrying`, `pickup_event`, `delivery_event`, `nest_x/y`,
  `target_<id>_x/y`
- Navigation: `q_shape`, `q_total`, `score`, `corridor_width`, `goal_x/y`,
  `goal_threshold`
- Rescue: `role`, `rescued_count`, `chain_connected`, `base_x/y`,
  `target_0_x/y`, `comm_view`
- 물리 파라미터: `w_*`, `k_diff`, `lambda_*`, `beta`, `alpha` 또는 `Param_*`
- 실행 물리 진단: `pinn/loss_micro`, `pinn/loss_macro`,
  `pinn/loss_total_raw`, `pinn/executed_edm_error`, `pinn/macro_residual_rms`,
  `pinn/mode_exact_*`

한 `(step, agent_id)`에 행이 둘 이상이면 `summary.json`의
`duplicate_agent_steps`로 보고한다. 숫자 결측치도 집계하되 원본을 자동 보정하지
않는다.

### 3.2 Training/evaluation CSV

`step`과 하나 이상의 숫자 metric 열이 필요하다. `pinn/`, `Param_`, reward,
collision, delivery, rescued, score, loss 등의 이름을 그룹화한다. 평가 전용 로그는
모든 행의 `step=0`일 수 있으므로 그래프 x축에 record index를 사용하되 JSON에는
원래 step을 보존한다.

## 4. 산출물

### Trajectory 입력

- `summary.json`: 과제, 행/agent/step 수, 데이터 품질, 과제 지표
- `per_step.csv`: agent 행을 swarm-level 시계열로 집계한 데이터
- `trajectories.png`: 시작/끝과 anchor/target을 포함한 로봇 궤적
- `swarm_dynamics.png`: 생존·충돌, 작업 진행, phase 비율, 로봇 간 거리
- `physical_parameters.png`: advection, diffusion/reaction, geometry를 단위별 subplot으로 분리
- `parameter_correlations.png`: 물리 파라미터와 outcome의 Pearson 상관

마지막 두 PNG는 관련 열이 있을 때만 생성된다.

### Training 입력

- `summary.json`: 각 숫자 열의 first/last/min/max/mean과 데이터 품질
- `learning_curves.png`: reward/score/성공 관련 곡선
- `physical_parameters.png`: 학습된 물리 파라미터
- `training_stability.png`: policy/value/entropy/PINN loss
- `task_diagnostics.png`: collision, error, delivery, rescue 등 과제 진단
- `parameter_correlations.png`: 파라미터와 outcome 상관

데이터에 해당 열이 없는 선택 그래프는 빈 파일을 만들지 않는다.

exact loss를 비교할 때 최적화에 들어간 soft-capped `pinn_loss`만 보지 않는다.
cap 전 micro/macro 원시 loss와 EDM error/residual RMS를 함께 그려야 “loss scale이
포화되어 좋아 보이는 현상”과 실제 물리 residual 감소를 구분할 수 있다. mode flag는
한 run이 어떤 경로를 탔는지 검증하는 provenance 열이며 성능 metric으로 평균내지
않는다.

## 5. 지표 정의

공통 구현은 [`experiments/metrics.py`](../experiments/metrics.py)에
있다.

### 실행 smoothness

```text
J_smooth = sum ||c_i(t) - c_i(t-1)||_1 / (N × (T-1))
```

식 S174 그대로 left/right wheel command의 시간 변화를 계산하며 작을수록 좋다.
`is_alive`가 있으면 양쪽 시점에 살아 있는 robot-transition만 분모에 넣어 고장 뒤
zero command를 가짜 jerk로 세지 않는다. 어떤 command 열을 사용했는지도
`summary.json`에 남긴다.

### 안전성

```text
collision rate = collision indicator의 평균
```

이벤트 로그가 “충돌 중인 frame”인지 “새 충돌 event”인지에 따라 의미가 달라질
수 있으므로 CSV schema에 반드시 기록해야 한다.

### Foraging

```text
efficiency = delivered resources / (number of robots × episode duration)
transport economy = delivered resources × reference trip distance / total robot path length
```

throughput와 efficiency를 같이 보고해야 robot 수나 horizon 증가가 만든 착시를
줄일 수 있다. transport economy는 nest와 resource 사이의 기준 거리 및 실제
로봇 궤적이 모두 있을 때만 계산한다.

### Navigation formation error

`formation_error`는 목표점 집합과 관측점 집합의 centroid를 제거하고 orthogonal
Procrustes 정렬 후 RMS 오차를 계산한다. 따라서 전역 translation과 rotation에
불변이다. scale은 보존하므로 대형이 과도하게 팽창·수축하면 오차에 반영된다.

현재 generic trajectory reporter는 target correspondence가 없는 legacy CSV에는
이 값을 자동 계산하지 않는다. 목표 slot 또는 canonical formation 좌표를
manifest에 추가한 실험에서 직접 metric API로 계산한다.

`navigation_time`은 식 S181에 따라 swarm centroid가 `goal_threshold` 안에 처음
들어온 실제 step을 반환한다. 실패하면 run에서 기록한 `episode_max_steps`를
반환하며, threshold나 `T_max`가 CSV에 없을 때 임의 기본값을 추측하지 않는다.
`navigation_time_censored`는 실패로 `T_max`를 받은 경우 `true`여서 마지막 step에
실제로 도착한 episode와 구분한다.

### Rescue connectivity

통신 반경 안의 base--relay--responder--target graph를 만들고 Laplacian의 두 번째
고윳값 `λ2`를 algebraic connectivity로 사용한다. `chain_connected`로 성공 여부와
최초 connectivity step을 보고하고, `relay_lambda2_mean/max`로 연결의 강도를
보조 진단한다.

식 S183--S185에 맞춰 trial success는 episode 중 한 번이라도 `λ2>0`인지로 정하고,
실패 trial의 time-to-connectivity는 `None`으로 버리지 않고 `T_max`로 censor한다.
`summary.json`의 `time_to_connectivity_censored`가 이 실패 penalty 여부를 표시한다.

## 6. 그래프 해석 원칙

- 학습 곡선 한 개만으로 결론 내리지 않고 seed별 원자료와 집계 신뢰구간을 보존한다.
- 서로 다른 단위의 파라미터를 같은 y축에 겹치지 않는다.
- correlation은 인과가 아니며, parameter/outcome 후보가 18개를 넘으면 가독성을
  위해 분산이 있는 앞 18개만 표시한다.
- smoothing을 적용한 그래프를 추가할 경우 raw curve도 함께 남기고 window를
  manifest에 기록한다.
- 실패 episode를 버리지 않는다. 성공 조건과 censoring 규칙을 JSON/manifest에
  명시한다.

## 7. 논문 재현에 추가로 필요한 집계

다중 seed 비교는 manifest를 입력으로 실행한다.

```csv
variant,seed,path
mappo,1,runs/mappo-seed1.csv
mappo,2,runs/mappo-seed2.csv
physics,1,runs/physics-seed1.csv
physics,2,runs/physics-seed2.csv
```

상대 `path`는 manifest 위치를 기준으로 해석한다. 같은 `(variant, seed)`는 하나의
독립 run만 가리켜야 한다.

```bash
python -m analysis.compare \
  --manifest experiments/manifest.csv \
  --output reports/comparison \
  --baseline mappo \
  --paper-values experiments/paper-values.csv \
  --metric average_episode_rewards \
  --metric pinn/loss_macro
```

논문 기준값 파일은 다음처럼 원 run과 분리해 보존한다.

```csv
metric,paper_value
average_episode_rewards,12.4
pinn/loss_macro,0.18
```

`--metric`을 생략하면 모든 run에 공통으로 존재하는 숫자 열을 집계한다. 서로 다른
logging 간격은 각 run의 실제 관측 범위 안에서만 선형 보간하며 외삽하지 않는다.
seed/run을 독립 단위로 variant별 평균, 표본 표준편차, 양측 Student-t 95% CI를
계산한다. `n=1`에서는 추정할 수 없는 CI를 억지로 0 폭으로 만들지 않고 비워 둔다.
legacy evaluation CSV처럼 모든 `step`이 같은 여러 행은 행 순서를 분석용 축으로
사용한다.

산출물은 다음과 같다.

- `aggregate.csv`: variant/metric/step별 `n`, mean, std, CI
- `final_metrics.csv`: 각 source run의 마지막 실제 관측값
- `statistical_comparisons.csv`: `--baseline` 지정 시 candidate-baseline 평균 차이,
  재현 가능한 bootstrap 95% CI, t-test, effect size
- `paper_comparison.csv`: `--paper-values` 지정 시 variant별 reproduced mean과
  paper value의 signed/absolute/relative gap
- `summary.json`: seed 목록, 계산 규칙, 최종 통계
- `outcomes_comparison*.png`, `losses_comparison*.png`,
  `parameters_comparison*.png`, `diagnostics_comparison*.png`
- `final_metric_distributions.png`: box plot과 모든 seed 점

동일 seed 집합이면 paired t-test와 Cohen `dz`, seed 집합이 다르면 모든 run을
보존한 Welch t-test와 small-sample corrected Hedges `g`를 사용한다. bootstrap은
기본 10,000회와 seed 0으로 결정적이며 CLI에서 바꿀 수 있다. 모든 차이는
`candidate - baseline`이고 metric의 “클수록 좋음/작을수록 좋음” 방향은 도구가
추측하지 않는다. p-value 하나만으로 결론 내리지 않고 raw seed 점, CI, effect
size를 함께 본다.

논문 표를 완전히 재현하려면 이 집계 위에 다음 단계가 더 필요하다.

1. scenario/variant/seed별 동일 schema 검증
2. seed별 episode metric 계산
3. metric별 성공/실패 episode 및 censoring 규칙의 사전 고정
4. 논문이 사용한 정확한 seed 수와 원자료 확인
5. 여러 metric 검정 시 multiplicity 정책과 practical significance threshold 명시

집계, bootstrap, 검정 선택, effect size, paper gap 표와 그래프는 구현되어 있다.
남은 항목은 숫자를 계산하는 코드가 아니라 실험 protocol에서 확정해야 할
연구 설계다. 단일 run 그래프를 다중 seed 통계로 오해하지 않도록 보고서 제목과
실험 기록에 구분한다.
