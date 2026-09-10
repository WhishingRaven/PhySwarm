# 학습·평가·추론 재현 절차

## 1. 환경

지원 기준은 Apple Silicon macOS, Python 3.13, Webots R2025a다.

```bash
conda env create -f environment.yml
conda activate physwarm
python -m pip check
python -m pytest -q
```

Webots의 기본 위치는 `/Applications/Webots.app`이다. 다른 위치라면
`WEBOTS_HOME`을 application bundle로 지정한다. Python launcher가 Webots API path,
native library path와 MPS fallback을 자식 process에 설정하므로 별도 shell source는
필요 없다.

## 2. 단일 Python 실행 계약

세 진입점은 같은 인자를 사용한다.

```bash
python train.py SCENARIO --webots MODE [MAPPO options]
python evaluate.py SCENARIO --webots MODE --model-dir PATH [--episodes N]
python infer.py SCENARIO --webots MODE --model-dir PATH
```

- `SCENARIO`: `foraging`, `navigation`, `rescue`
- `MODE`: `fast`, `realtime`, `existing`
- `fast`: batch/no-rendering Webots를 실행하고 종료까지 소유한다.
- `realtime`: GUI realtime Webots를 실행하고 종료까지 소유한다.
- `existing`: 이미 실행 중인 `ipc://1234/supervisor` endpoint에 연결한다.

`--controller-url ipc://PORT/NAME`, `--webots-home PATH`,
`--startup-timeout SECONDS`로 adapter 설정을 바꾼다. 나머지 옵션은 공용
`learning.rmappo.config`가 검증한다. 오타나 소유되지 않은 옵션은 즉시 실패한다.

## 3. 학습

```bash
python train.py foraging --webots fast
python train.py navigation --webots fast --seed 19 --experiment_name paper_joint \
  --parameter_contract paper --phase_contract paper \
  --physics_loss_mode exact_joint
python train.py rescue --webots realtime --physics_loss_mode exact_micro
```

새 산출물은 아래에 생성된다.

```text
results/<scenario>/<env>/<algorithm>/<experiment>/runN/
├── config.csv
├── logs/
└── models/
```

기존 체크인 model과 과거 결과는 `controllers/Swarm_*/`에 그대로 보존된다.

중단된 실행은 model directory를 지정해 재개한다.

```bash
python train.py foraging --webots fast --resume \
  --model-dir results/foraging/Dynamical_system/mappo/debug/run1/models
```

`checkpoint.pt`에는 환경 step, episode/update 위치, optimizer와 ValueNorm 상태가
들어간다. actor/critic network는 `models/policy_0/*.pt`에 저장된다.

## 4. 평가와 추론

평가는 여러 episode를 집계하고, 추론은 기본적으로 deterministic 한 episode를
실행한다.

```bash
python evaluate.py foraging --webots fast \
  --model-dir controllers/Swarm_Foraging/models --episodes 60

python infer.py navigation --webots realtime \
  --model-dir controllers/Swarm_Navigation/models
```

두 명령은 `action distribution.mean`을 사용하며 모델을 수정하지 않는다.

## 5. 빠른 smoke test

한 번의 짧은 rollout/update만 검증하려면 다음처럼 기본값을 덮어쓴다.

```bash
python train.py foraging --webots fast \
  --episode_length 4 --num_env_steps 4 \
  --buffer_size 1 --num_mini_batch 1 \
  --train_interval_episode 1 --data_chunk_length 4 \
  --use_eval --use_save --log_interval 999999
```

`--use_eval`과 `--use_save`는 기존 argparse 의미를 보존해 각각 주기 평가와 저장을
끄는 flag다. 실제 명령을 실행하지 않고 조합만 검사하려면 `--dry-run`을 쓴다.

## 6. 궤적과 분석

궤적 기록은 명시적으로 켠다. 현재 schema에 `env_id`가 없으므로
`--n_rollout_threads 1`만 허용하며 기존 파일을 덮어쓰지 않는다.

```bash
python evaluate.py rescue --webots fast \
  --model-dir controllers/Swarm_Rescue/models --episodes 5 \
  --save_trajectory --trajectory_output data/rescue_eval

python -m analysis --input data/rescue_eval --output reports/rescue --scenario rescue
python -m analysis.compare --manifest experiments/example_manifest.csv \
  --output reports/comparison
```

## 7. 재현 기록

장시간 결과에는 최소한 commit, Webots/Python/PyTorch 버전, device, scenario,
seed, parameter/phase contract, physics loss mode, 전체 config.csv, checkpoint와
trajectory를 함께 보존한다. 짧은 smoke run은 수치 재현 실험 결과로 보고하지 않는다.
