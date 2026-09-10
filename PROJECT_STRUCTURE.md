# PhySwarm 프로젝트 구조

상세 구조 문서는 [`docs/architecture.md`](docs/architecture.md)로 이동했습니다.
이 파일은 기존 링크와 북마크의 호환성을 위해 유지합니다.

현재 최상위 책임은 다음과 같이 나뉩니다.

```text
physics/       Webots 비의존 Macro-ADR, Micro-EDM, 파라미터 사상
learning/      공용 rMAPPO, rollout, 미분 가능 물리 손실
tasks/         과제 계약과 기본 실행 profile
adapters/      서로 분리된 Webots 및 Gymnasium 경계
app/           학습·평가·추론 Python 실행 수명주기
experiments/   궤적 저장 계약과 평가 지표
analysis/      CSV 보고서와 multi-seed 비교 CLI
controllers/   Webots가 직접 시작하는 e-puck controller와 보존된 산출물
worlds/        Webots world 생성기와 생성 결과
tests/         런타임·checkpoint·수식·지표·분석 회귀 테스트
docs/          구조, 논문 감사, 재현, 실험, 확장 문서
```

구현 상태를 판단할 때는
[`docs/paper-implementation-audit.md`](docs/paper-implementation-audit.md)를,
실제 실행 명령은 [`docs/reproduction.md`](docs/reproduction.md)를 기준으로
사용하십시오.
