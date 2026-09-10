"""학습 로그와 로봇 궤적을 일관된 보고서로 변환한다.

비교 모듈은 ``python -m analysis.compare``로도 실행된다. package import
단계에서 해당 모듈을 미리 불러오면 ``runpy``가 이중 실행 경고를 내므로 비교 API는
PEP 562의 lazy attribute로 노출한다.
"""

from typing import Any

from analysis.report import AnalysisResult, analyze_csv

__all__ = [
    "AnalysisResult",
    "ComparisonResult",
    "RunSpec",
    "analyze_csv",
    "compare_training_runs",
    "load_manifest",
    "load_paper_values",
]


def __getattr__(name: str) -> Any:
    """비교 API를 처음 요청할 때만 import해 module CLI 실행과 충돌하지 않는다."""

    if name in {
        "ComparisonResult",
        "RunSpec",
        "compare_training_runs",
        "load_manifest",
        "load_paper_values",
    }:
        from analysis import compare

        return getattr(compare, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
