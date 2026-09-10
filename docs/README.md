# PhySwarm 문서 안내

이 디렉터리는 PhySwarm을 **실행하는 방법**, **현재 구현이 논문과 어디까지
일치하는지**, **기존 과제를 유지하면서 방법론을 교체하는 방법**을 서로 다른
문서로 분리한다. 루트 README는 빠른 시작만 담당하고, 아래 문서가 상세 설명의
기준(source of truth)이다.

| 문서 | 먼저 읽을 때 | 답하는 질문 |
|---|---|---|
| [architecture.md](architecture.md) | 코드를 수정하기 전 | 환경, 시뮬레이션, 학습, 물리식, 분석의 경계는 어디인가? |
| [paper-implementation-audit.md](paper-implementation-audit.md) | 논문 재현성을 판단할 때 | 논문의 각 수식·실험 요소가 실제로 구현되고 연결되어 있는가? |
| [reproduction.md](reproduction.md) | 학습·평가를 실행할 때 | 어떤 환경과 명령, seed, 산출물이 필요한가? |
| [experiments.md](experiments.md) | CSV를 비교·시각화할 때 | 어떤 지표와 그래프를 어떤 입력에서 만드는가? |
| [extending-methodology.md](extending-methodology.md) | 새 알고리즘을 붙일 때 | 과제 정의를 보존한 채 정책·물리 모델·손실을 어떻게 바꾸는가? |

## 현재 상태를 읽는 법

문서의 구현 상태는 다음 의미로 사용한다.

- **연결됨**: Webots 실행 또는 학습 경로에서 실제 호출되며 테스트가 있다.
- **부분 구현**: 유사한 계산이 실행되지만 논문의 정의나 입력이 완전하지 않다.
- **독립 구현**: 공통 모듈과 단위 테스트는 있으나 기존 시나리오 학습기에 아직
  연결되지 않았다.
- **미구현**: 코드 또는 필요한 데이터 경로가 없다.

가장 중요한 결론은 간단하다. 현재 저장소는 공용 rMAPPO 실행 코어로 세 Webots 과제를
실행하며, 별도 transition side-channel에서 phase별 KDE 시공간장을 구성해 exact
micro/macro/joint loss를 actor 학습에 선택적으로 연결한다. 다만 논문 내부의
observation/action 명세 충돌, world·플랫폼 차이, 미실행 장시간 multi-seed
ablation이 남아 있다. 따라서 현재 체크포인트나 짧은 smoke 결과를 논문의 완전
재현 결과라고 표현해서는 안 된다.

## 가장 짧은 검증 경로

```bash
conda run -n physwarm python -m pip check
conda run -n physwarm python -m pytest -q
conda run -n physwarm python -m analysis --help
```

물리 코어의 공개 진입점은 [`physics`](../physics), rollout 기반
미분 가능 손실은
[`learning/trajectory_loss.py`](../learning/trajectory_loss.py),
공통 지표와 보고서는 각각 [`experiments`](../experiments)와
[`analysis`](../analysis)에 있다.
