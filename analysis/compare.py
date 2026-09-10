"""여러 seed와 방법 variant의 학습 CSV를 공정하게 집계한다.

단일 run의 멋진 곡선은 알고리즘 비교 근거가 아니다. 이 모듈은 명시적인
manifest에서 run identity를 읽고, 서로 다른 logging step을 보간해 variant별
평균과 Student-t 95% 신뢰구간을 계산한다. 원본 CSV에는 쓰지 않으며 모든 중간
집계를 CSV/JSON으로 남겨 그림의 숫자를 역추적할 수 있게 한다.

Manifest schema
---------------
``variant, seed, path`` 세 열이 필수다. ``path``가 상대 경로면 manifest가 있는
디렉터리를 기준으로 해석한다. 같은 ``(variant, seed)`` 조합은 하나의 독립 run만
가리켜야 하므로 중복을 오류로 처리한다.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_ind, ttest_rel

# report.py와 같은 headless renderer 계약을 독립 CLI에서도 유지한다.
_MPL_CACHE = Path(tempfile.gettempdir()) / "physwarm-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MANIFEST_COLUMNS = {"variant", "seed", "path"}
OUTCOME_TOKENS = (
    "reward",
    "score",
    "success",
    "efficiency",
    "throughput",
    "delivery",
    "rescued",
    "collision",
    "error",
    "connect",
)
LOSS_TOKENS = ("loss", "entropy", "grad_norm", "ratio")
PARAMETER_PREFIXES = ("w_", "k_diff", "lambda_", "beta", "alpha")


@dataclass(frozen=True)
class RunSpec:
    """비교 집합 안에서 하나의 통계적으로 독립적인 학습 run."""

    variant: str
    seed: int
    path: Path


@dataclass(frozen=True)
class ComparisonResult:
    """다중 run 분석이 생성한 추적 가능한 산출물."""

    output_directory: Path
    generated_files: tuple[Path, ...]
    summary: dict[str, object]


def load_manifest(path: str | Path) -> tuple[RunSpec, ...]:
    """CSV manifest를 검증하고 모든 run path를 절대 경로로 해석한다."""

    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    missing = MANIFEST_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Manifest must contain at least one run")
    if frame[["variant", "seed"]].duplicated().any():
        duplicates = frame.loc[
            frame[["variant", "seed"]].duplicated(keep=False),
            ["variant", "seed"],
        ].to_dict("records")
        raise ValueError(f"Duplicate (variant, seed) entries: {duplicates}")

    runs: list[RunSpec] = []
    for row in frame.itertuples(index=False):
        variant = str(row.variant).strip()
        if not variant:
            raise ValueError("variant names must not be empty")
        try:
            seed = int(row.seed)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid seed for variant {variant!r}: {row.seed!r}") from error
        run_path = Path(str(row.path)).expanduser()
        if not run_path.is_absolute():
            run_path = manifest_path.parent / run_path
        run_path = run_path.resolve()
        if not run_path.is_file():
            raise FileNotFoundError(f"Run CSV does not exist: {run_path}")
        runs.append(RunSpec(variant=variant, seed=seed, path=run_path))
    return tuple(runs)


def load_paper_values(path: str | Path) -> dict[str, float]:
    """`metric,paper_value` CSV를 유한한 이름→값 mapping으로 읽는다.

    논문 표의 숫자는 run manifest와 분리한다. 그래야 새 실험 CSV를 추가하는
    과정에서 reference 값이 은근히 바뀌지 않고, 어떤 원문 숫자와 비교했는지
    입력 파일 자체로 추적할 수 있다.
    """

    source = Path(path).expanduser().resolve()
    frame = pd.read_csv(source)
    required = {"metric", "paper_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Paper-value CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Paper-value CSV must contain at least one metric")

    names = frame["metric"].astype(str).str.strip()
    if (names == "").any():
        raise ValueError("Paper-value metric names must not be empty")
    if names.duplicated().any():
        duplicates = sorted(set(names[names.duplicated(keep=False)].tolist()))
        raise ValueError(f"Duplicate paper-value metrics: {duplicates}")
    values = pd.to_numeric(frame["paper_value"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("paper_value entries must be finite numbers")
    return {
        str(metric): float(value)
        for metric, value in zip(names, values.astype(float))
    }


def _read_training_frame(run: RunSpec) -> pd.DataFrame:
    """한 run을 숫자 step 시계열로 정규화하되 결과값을 임의 보정하지 않는다."""

    frame = pd.read_csv(run.path)
    if "step" not in frame:
        raise ValueError(f"Run CSV is missing 'step': {run.path}")
    if frame.empty:
        raise ValueError(f"Run CSV is empty: {run.path}")

    normalized = frame.copy()
    normalized["step"] = pd.to_numeric(normalized["step"], errors="coerce")
    if normalized["step"].isna().any():
        raise ValueError(f"Run contains non-numeric step values: {run.path}")
    if len(normalized) > 1 and normalized["step"].nunique() == 1:
        # 일부 legacy evaluation logger는 실제 순차 record에도 step=0만 쓴다.
        # 이때 groupby로 한 점으로 접으면 학습/평가 추세가 사라지므로 단일-run
        # reporter와 같은 규칙으로 record index를 분석용 축으로 사용한다.
        normalized["step"] = np.arange(len(normalized), dtype=np.float64)
    numeric = [
        column
        for column in normalized.columns
        if column != "step" and pd.api.types.is_numeric_dtype(normalized[column])
    ]
    if not numeric:
        raise ValueError(f"Run has no numeric metrics: {run.path}")

    # 동일 step의 여러 log record는 seed 수를 부풀리지 않도록 run 안에서 먼저 평균낸다.
    return (
        normalized[["step", *numeric]]
        .groupby("step", as_index=False, sort=True)
        .mean(numeric_only=True)
    )


def _common_metrics(
    frames: Sequence[pd.DataFrame],
    requested: Sequence[str] | None,
) -> list[str]:
    """모든 run에 실제 숫자로 존재하는 공정 비교 metric을 선택한다."""

    shared = set(frames[0].columns) - {"step"}
    for frame in frames[1:]:
        shared.intersection_update(set(frame.columns) - {"step"})

    if requested:
        missing = [metric for metric in requested if metric not in shared]
        if missing:
            raise ValueError(
                "Requested metrics are not numeric columns in every run: "
                f"{missing}"
            )
        return list(dict.fromkeys(requested))

    # auto mode에서도 parameter만 있는 로그를 분석할 수 있도록 shared metric을 모두
    # 유지하되, 사람이 예상할 수 있는 lexical order를 사용한다.
    if not shared:
        raise ValueError("No numeric metric is shared by every run")
    return sorted(shared)


def _interpolate_without_extrapolation(
    frame: pd.DataFrame,
    metric: str,
    grid: np.ndarray,
) -> np.ndarray:
    """run이 실제로 관측한 step 범위 안에서만 선형 보간한다."""

    valid = frame[["step", metric]].dropna()
    result = np.full(grid.shape, np.nan, dtype=np.float64)
    if valid.empty:
        return result
    steps = valid["step"].to_numpy(dtype=np.float64)
    values = valid[metric].to_numpy(dtype=np.float64)
    in_range = (grid >= steps[0]) & (grid <= steps[-1])
    if len(steps) == 1:
        # 단일 관측을 전체 horizon으로 복제하면 거짓 곡선을 만든다. 같은 step만 채운다.
        result[np.isclose(grid, steps[0])] = values[0]
    else:
        result[in_range] = np.interp(grid[in_range], steps, values)
    return result


def _mean_and_ci(values: np.ndarray) -> tuple[np.ndarray, ...]:
    """열별 유효 run 수와 sample standard deviation 기반 t interval을 계산한다."""

    count = np.sum(np.isfinite(values), axis=0).astype(np.int64)
    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    std = np.full(values.shape[1], np.nan, dtype=np.float64)
    low = np.full(values.shape[1], np.nan, dtype=np.float64)
    high = np.full(values.shape[1], np.nan, dtype=np.float64)

    for index, n_value in enumerate(count):
        column = values[:, index]
        column = column[np.isfinite(column)]
        if n_value == 0:
            continue
        mean[index] = float(np.mean(column))
        if n_value == 1:
            # 불확실성을 추정할 수 없으므로 CI를 만들지 않는다.
            continue
        std[index] = float(np.std(column, ddof=1))
        critical = float(student_t.ppf(0.975, df=n_value - 1))
        half_width = critical * std[index] / np.sqrt(n_value)
        low[index] = mean[index] - half_width
        high[index] = mean[index] + half_width
    return count, mean, std, low, high


def _bootstrap_difference_ci(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    paired: bool,
    samples: int,
    generator: np.random.Generator,
) -> tuple[float, float]:
    """seed를 통계 단위로 재표집한 평균 차이의 percentile 95% CI다.

    같은 seed 집합이면 seed pair 전체를 함께 재표집하고, 그렇지 않으면 두
    variant를 독립적으로 재표집한다. 표본 하나의 고정값을 신뢰구간처럼 보이지
    않게 양쪽에 최소 두 표본이 없으면 NaN을 반환한다.
    """

    if paired:
        if candidate.size < 2:
            return np.nan, np.nan
        differences = candidate - baseline
        indices = generator.integers(
            0, differences.size, size=(samples, differences.size)
        )
        estimates = differences[indices].mean(axis=1)
    else:
        if candidate.size < 2 or baseline.size < 2:
            return np.nan, np.nan
        candidate_indices = generator.integers(
            0, candidate.size, size=(samples, candidate.size)
        )
        baseline_indices = generator.integers(
            0, baseline.size, size=(samples, baseline.size)
        )
        estimates = (
            candidate[candidate_indices].mean(axis=1)
            - baseline[baseline_indices].mean(axis=1)
        )
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def _paired_effect_size(differences: np.ndarray) -> float:
    """paired comparison의 Cohen dz; 분산이 없으면 정의되지 않는다."""

    if differences.size < 2:
        return np.nan
    standard_deviation = float(np.std(differences, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return np.nan
    return float(np.mean(differences) / standard_deviation)


def _unpaired_effect_size(candidate: np.ndarray, baseline: np.ndarray) -> float:
    """서로 다른 seed 집합에 쓰는 small-sample corrected Hedges g다."""

    if candidate.size < 2 or baseline.size < 2:
        return np.nan
    degrees_of_freedom = candidate.size + baseline.size - 2
    pooled_variance = (
        (candidate.size - 1) * np.var(candidate, ddof=1)
        + (baseline.size - 1) * np.var(baseline, ddof=1)
    ) / degrees_of_freedom
    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        return np.nan
    cohen_d = (float(np.mean(candidate)) - float(np.mean(baseline))) / np.sqrt(
        pooled_variance
    )
    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)
    return float(correction * cohen_d)


def _final_statistical_comparisons(
    final_metrics: pd.DataFrame,
    *,
    variants: Sequence[str],
    metrics: Sequence[str],
    baseline_variant: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """각 candidate의 마지막 관측값을 지정 baseline과 검정한다.

    두 variant의 seed 집합이 정확히 같을 때만 paired test를 쓴다. 일부 seed만
    겹칠 때 편의상 표본을 버리지 않고 모든 run을 사용한 Welch test로 명시한다.
    차이는 항상 ``candidate - baseline``이며 metric의 좋고 나쁜 방향을 추측하지
    않는다.
    """

    records: list[dict[str, object]] = []
    generator = np.random.default_rng(bootstrap_seed)
    candidates = [variant for variant in variants if variant != baseline_variant]
    for metric in metrics:
        metric_frame = final_metrics[final_metrics["metric"] == metric]
        baseline_frame = metric_frame[
            metric_frame["variant"] == baseline_variant
        ][["seed", "value"]].sort_values("seed")
        for candidate_variant in candidates:
            candidate_frame = metric_frame[
                metric_frame["variant"] == candidate_variant
            ][["seed", "value"]].sort_values("seed")
            if baseline_frame.empty or candidate_frame.empty:
                continue

            baseline_seeds = set(baseline_frame["seed"].astype(int))
            candidate_seeds = set(candidate_frame["seed"].astype(int))
            paired = baseline_seeds == candidate_seeds and len(baseline_seeds) >= 2
            if paired:
                aligned = baseline_frame.merge(
                    candidate_frame,
                    on="seed",
                    suffixes=("_baseline", "_candidate"),
                    validate="one_to_one",
                ).sort_values("seed")
                baseline_values = aligned["value_baseline"].to_numpy(dtype=float)
                candidate_values = aligned["value_candidate"].to_numpy(dtype=float)
                differences = candidate_values - baseline_values
                # 모든 paired 차이가 같으면 standard error가 0이라 t statistic은
                # 정의되지 않는다. SciPy 경고나 가짜 유의확률 대신 NaN으로 남긴다.
                if float(np.std(differences, ddof=1)) > 0.0:
                    test_result = ttest_rel(candidate_values, baseline_values)
                    statistic = float(test_result.statistic)
                    p_value = float(test_result.pvalue)
                else:
                    statistic = np.nan
                    p_value = np.nan
                effect_name = "cohen_dz"
                effect_size = _paired_effect_size(differences)
                comparison = "paired-by-seed"
                pair_count = len(aligned)
            else:
                baseline_values = baseline_frame["value"].to_numpy(dtype=float)
                candidate_values = candidate_frame["value"].to_numpy(dtype=float)
                standard_error_squared = (
                    np.var(candidate_values, ddof=1) / candidate_values.size
                    + np.var(baseline_values, ddof=1) / baseline_values.size
                    if baseline_values.size >= 2 and candidate_values.size >= 2
                    else 0.0
                )
                if standard_error_squared > 0.0:
                    test_result = ttest_ind(
                        candidate_values,
                        baseline_values,
                        equal_var=False,
                    )
                    statistic = float(test_result.statistic)
                    p_value = float(test_result.pvalue)
                else:
                    statistic = np.nan
                    p_value = np.nan
                effect_name = "hedges_g"
                effect_size = _unpaired_effect_size(
                    candidate_values, baseline_values
                )
                comparison = "unpaired-welch"
                pair_count = len(baseline_seeds.intersection(candidate_seeds))

            difference = float(np.mean(candidate_values) - np.mean(baseline_values))
            baseline_mean = float(np.mean(baseline_values))
            bootstrap_low, bootstrap_high = _bootstrap_difference_ci(
                candidate_values,
                baseline_values,
                paired=paired,
                samples=bootstrap_samples,
                generator=generator,
            )
            records.append(
                {
                    "baseline": baseline_variant,
                    "candidate": candidate_variant,
                    "metric": metric,
                    "comparison": comparison,
                    "n_baseline": int(baseline_values.size),
                    "n_candidate": int(candidate_values.size),
                    "n_shared_seeds": int(pair_count),
                    "baseline_mean": baseline_mean,
                    "candidate_mean": float(np.mean(candidate_values)),
                    "mean_difference": difference,
                    "relative_difference_percent": (
                        100.0 * difference / abs(baseline_mean)
                        if baseline_mean != 0.0
                        else np.nan
                    ),
                    "bootstrap_ci95_low": bootstrap_low,
                    "bootstrap_ci95_high": bootstrap_high,
                    "test": "paired-t" if paired else "welch-t",
                    "test_statistic": statistic,
                    "p_value": p_value,
                    "effect_size_name": effect_name,
                    "effect_size": effect_size,
                }
            )
    return pd.DataFrame.from_records(records)


def _paper_gap_table(
    final_metrics: pd.DataFrame,
    paper_values: Mapping[str, float],
    variants: Sequence[str],
) -> pd.DataFrame:
    """각 variant의 final mean과 사용자가 제공한 논문 기준값의 gap을 만든다."""

    records: list[dict[str, object]] = []
    for metric, paper_value in paper_values.items():
        metric_frame = final_metrics[final_metrics["metric"] == metric]
        for variant in variants:
            values = metric_frame.loc[
                metric_frame["variant"] == variant, "value"
            ].to_numpy(dtype=float)
            if values.size == 0:
                continue
            reproduced = float(np.mean(values))
            signed_gap = reproduced - float(paper_value)
            records.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n": int(values.size),
                    "paper_value": float(paper_value),
                    "reproduced_mean": reproduced,
                    "signed_gap": signed_gap,
                    "absolute_gap": abs(signed_gap),
                    "relative_gap_percent": (
                        100.0 * signed_gap / abs(float(paper_value))
                        if paper_value != 0.0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _metric_category(metric: str) -> str:
    """열 이름을 outcome/loss/parameter/diagnostic 시각 그룹으로 분류한다."""

    lowered = metric.lower()
    leaf = metric.split("/")[-1].lower()
    if metric.startswith("Param_") or leaf.startswith(PARAMETER_PREFIXES):
        return "parameters"
    if any(token in lowered for token in LOSS_TOKENS):
        return "losses"
    if any(token in lowered for token in OUTCOME_TOKENS):
        return "outcomes"
    return "diagnostics"


def _save_curve_pages(
    aggregate: pd.DataFrame,
    metrics: Sequence[str],
    output_directory: Path,
    generated: list[Path],
) -> None:
    """범주별 최대 6개 subplot으로 mean/95% CI 곡선을 저장한다."""

    variants = list(dict.fromkeys(aggregate["variant"].astype(str)))
    colors = plt.get_cmap("tab10")
    for category in ("outcomes", "losses", "parameters", "diagnostics"):
        category_metrics = [m for m in metrics if _metric_category(m) == category]
        for page_index, page_start in enumerate(range(0, len(category_metrics), 6), start=1):
            page_metrics = category_metrics[page_start : page_start + 6]
            if not page_metrics:
                continue
            rows = int(np.ceil(len(page_metrics) / 2))
            figure, axes = plt.subplots(rows, 2, figsize=(12.0, 3.5 * rows), squeeze=False)
            for axis, metric in zip(axes.flat, page_metrics):
                for variant_index, variant in enumerate(variants):
                    series = aggregate[
                        (aggregate["variant"] == variant)
                        & (aggregate["metric"] == metric)
                    ].sort_values("step")
                    axis.plot(
                        series["step"],
                        series["mean"],
                        label=variant,
                        color=colors(variant_index % 10),
                        marker="o",
                        markersize=2.5,
                    )
                    ci_mask = series[["ci95_low", "ci95_high"]].notna().all(axis=1)
                    if ci_mask.any():
                        axis.fill_between(
                            series.loc[ci_mask, "step"],
                            series.loc[ci_mask, "ci95_low"],
                            series.loc[ci_mask, "ci95_high"],
                            color=colors(variant_index % 10),
                            alpha=0.14,
                        )
                axis.set_title(metric)
                axis.set_xlabel("training step")
                axis.set_ylabel("mean ± 95% t-CI")
                axis.grid(alpha=0.22)
                axis.legend(loc="best")
            for unused in axes.flat[len(page_metrics) :]:
                unused.set_visible(False)
            figure.tight_layout()
            suffix = "" if len(category_metrics) <= 6 else f"_{page_index}"
            path = output_directory / f"{category}_comparison{suffix}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            generated.append(path)


def _plot_final_distributions(
    final_metrics: pd.DataFrame,
    metrics: Sequence[str],
    output_directory: Path,
) -> Path:
    """각 seed의 최종값을 숨기지 않는 box + deterministic strip plot."""

    display_metrics = [
        metric for metric in metrics if _metric_category(metric) in {"outcomes", "losses"}
    ][:6]
    if not display_metrics:
        display_metrics = list(metrics[:6])
    variants = list(dict.fromkeys(final_metrics["variant"].astype(str)))
    rows = int(np.ceil(len(display_metrics) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(12.0, 3.6 * rows), squeeze=False)
    for axis, metric in zip(axes.flat, display_metrics):
        metric_frame = final_metrics[final_metrics["metric"] == metric]
        samples = [
            metric_frame.loc[metric_frame["variant"] == variant, "value"].to_numpy()
            for variant in variants
        ]
        axis.boxplot(samples, tick_labels=variants, showmeans=True)
        for variant_index, values in enumerate(samples, start=1):
            if len(values) == 0:
                continue
            # seed 순 정렬 후 고정 간격을 사용해 그림이 실행마다 흔들리지 않는다.
            offsets = np.linspace(-0.07, 0.07, len(values)) if len(values) > 1 else [0.0]
            axis.scatter(
                variant_index + np.asarray(offsets),
                values,
                color="#263238",
                s=18,
                alpha=0.72,
                zorder=3,
            )
        axis.set_title(metric)
        axis.set_ylabel("last observed value")
        axis.grid(axis="y", alpha=0.22)
    for unused in axes.flat[len(display_metrics) :]:
        unused.set_visible(False)
    figure.tight_layout()
    path = output_directory / "final_metric_distributions.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def compare_training_runs(
    runs: Iterable[RunSpec],
    output_directory: str | Path,
    *,
    metrics: Sequence[str] | None = None,
    baseline_variant: str | None = None,
    paper_values: Mapping[str, float] | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> ComparisonResult:
    """run을 집계하고 선택적으로 baseline/paper 기준과 통계 비교한다."""

    run_list = tuple(runs)
    if not run_list:
        raise ValueError("At least one run is required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    identities = [(run.variant, run.seed) for run in run_list]
    if len(identities) != len(set(identities)):
        raise ValueError("Every (variant, seed) pair must be unique")

    frames = [_read_training_frame(run) for run in run_list]
    selected_metrics = _common_metrics(frames, metrics)
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    aggregate_records: list[dict[str, object]] = []
    final_records: list[dict[str, object]] = []
    variants = list(dict.fromkeys(run.variant for run in run_list))
    if baseline_variant is not None:
        baseline_variant = baseline_variant.strip()
        if baseline_variant not in variants:
            raise ValueError(
                f"Unknown baseline variant {baseline_variant!r}; choose from {variants}"
            )
        if len(variants) < 2:
            raise ValueError("A baseline comparison requires at least two variants")

    normalized_paper_values: dict[str, float] = {}
    for name, value in (paper_values or {}).items():
        metric_name = str(name).strip()
        numeric_value = float(value)
        if not metric_name:
            raise ValueError("Paper-value metric names must not be empty")
        if not np.isfinite(numeric_value):
            raise ValueError("Paper values must be finite")
        if metric_name in normalized_paper_values:
            raise ValueError(f"Duplicate normalized paper metric: {metric_name!r}")
        normalized_paper_values[metric_name] = numeric_value
    unavailable_paper_metrics = sorted(
        set(normalized_paper_values).difference(selected_metrics)
    )
    if unavailable_paper_metrics:
        raise ValueError(
            "Paper-value metrics are not selected numeric columns in every run: "
            f"{unavailable_paper_metrics}"
        )
    for metric in selected_metrics:
        for run, frame in zip(run_list, frames):
            valid = frame[["step", metric]].dropna().sort_values("step")
            if valid.empty:
                continue
            final_records.append(
                {
                    "variant": run.variant,
                    "seed": run.seed,
                    "metric": metric,
                    "step": float(valid["step"].iloc[-1]),
                    "value": float(valid[metric].iloc[-1]),
                    "source": str(run.path),
                }
            )

        for variant in variants:
            variant_frames = [
                frame
                for run, frame in zip(run_list, frames)
                if run.variant == variant
            ]
            if not variant_frames:
                continue
            grid = np.unique(
                np.concatenate(
                    [frame.loc[frame[metric].notna(), "step"].to_numpy(dtype=float) for frame in variant_frames]
                )
            )
            if grid.size == 0:
                continue
            aligned = np.stack(
                [
                    _interpolate_without_extrapolation(frame, metric, grid)
                    for frame in variant_frames
                ],
                axis=0,
            )
            count, mean, std, low, high = _mean_and_ci(aligned)
            for index, step in enumerate(grid):
                aggregate_records.append(
                    {
                        "variant": variant,
                        "metric": metric,
                        "step": float(step),
                        "n": int(count[index]),
                        "mean": mean[index],
                        "std": std[index],
                        "ci95_low": low[index],
                        "ci95_high": high[index],
                    }
                )

    aggregate = pd.DataFrame.from_records(aggregate_records)
    final_frame = pd.DataFrame.from_records(final_records)
    aggregate_path = destination / "aggregate.csv"
    final_path = destination / "final_metrics.csv"
    aggregate.to_csv(aggregate_path, index=False)
    final_frame.to_csv(final_path, index=False)

    additional_paths: list[Path] = []
    if baseline_variant is not None:
        comparisons = _final_statistical_comparisons(
            final_frame,
            variants=variants,
            metrics=selected_metrics,
            baseline_variant=baseline_variant,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        comparisons_path = destination / "statistical_comparisons.csv"
        comparisons.to_csv(comparisons_path, index=False)
        additional_paths.append(comparisons_path)

    if normalized_paper_values:
        paper_gaps = _paper_gap_table(
            final_frame, normalized_paper_values, variants
        )
        paper_path = destination / "paper_comparison.csv"
        paper_gaps.to_csv(paper_path, index=False)
        additional_paths.append(paper_path)

    final_summary: dict[str, dict[str, object]] = {}
    for variant in variants:
        variant_summary: dict[str, object] = {}
        for metric in selected_metrics:
            values = final_frame.loc[
                (final_frame["variant"] == variant)
                & (final_frame["metric"] == metric),
                "value",
            ].to_numpy(dtype=float)
            if values.size == 0:
                continue
            entry: dict[str, object] = {
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else None,
                "ci95_low": None,
                "ci95_high": None,
            }
            if values.size > 1:
                critical = float(student_t.ppf(0.975, df=values.size - 1))
                half_width = critical * float(np.std(values, ddof=1)) / np.sqrt(values.size)
                entry["ci95_low"] = entry["mean"] - half_width
                entry["ci95_high"] = entry["mean"] + half_width
            variant_summary[metric] = entry
        final_summary[variant] = variant_summary

    summary: dict[str, object] = {
        "kind": "training-comparison",
        "run_count": len(run_list),
        "variants": {
            variant: sorted(run.seed for run in run_list if run.variant == variant)
            for variant in variants
        },
        "metrics": selected_metrics,
        "confidence_interval": "two-sided Student-t, 95%, run/seed as independent unit",
        "interpolation": "linear within each run's observed step range; no extrapolation",
        "final": final_summary,
    }
    if baseline_variant is not None:
        summary["statistical_comparison"] = {
            "baseline": baseline_variant,
            "difference": "candidate - baseline; metric direction is not inferred",
            "pairing": (
                "paired by seed only when the two variants have identical seed sets; "
                "otherwise all runs use Welch's independent-sample t-test"
            ),
            "bootstrap": (
                f"percentile 95% CI of mean difference; {bootstrap_samples} resamples; "
                f"seed {bootstrap_seed}"
            ),
            "effect_size": "Cohen dz when paired, small-sample corrected Hedges g otherwise",
        }
    if normalized_paper_values:
        summary["paper_values"] = normalized_paper_values
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    generated = [summary_path, aggregate_path, final_path, *additional_paths]
    _save_curve_pages(aggregate, selected_metrics, destination, generated)
    generated.append(
        _plot_final_distributions(final_frame, selected_metrics, destination)
    )
    return ComparisonResult(
        output_directory=destination,
        generated_files=tuple(generated),
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PhySwarm training variants across independent seeds."
    )
    parser.add_argument("--manifest", required=True, help="CSV with variant,seed,path")
    parser.add_argument("--output", required=True, help="directory for aggregate artifacts")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="metric shared by every run; repeat to select multiple (default: all shared)",
    )
    parser.add_argument(
        "--baseline",
        help=(
            "variant used as the statistical baseline; writes paired/Welch tests, "
            "effect sizes, and bootstrap confidence intervals"
        ),
    )
    parser.add_argument(
        "--paper-values",
        help="optional CSV with metric,paper_value reference columns",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="resamples for final mean-difference confidence intervals (default: 10000)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="deterministic bootstrap RNG seed (default: 0)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = load_manifest(args.manifest)
    paper_values = (
        load_paper_values(args.paper_values) if args.paper_values is not None else None
    )
    result = compare_training_runs(
        runs,
        args.output,
        metrics=args.metrics,
        baseline_variant=args.baseline,
        paper_values=paper_values,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"Compared {result.summary['run_count']} runs in {result.output_directory}")
    for path in result.generated_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
