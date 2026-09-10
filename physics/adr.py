"""Macro-ADR 보존 반응, 잔차와 논문 지표 E_ADR 구현."""

from __future__ import annotations

import numpy as np


def conservative_reaction_matrix(
    transition_rates: np.ndarray,
    transition_mask: np.ndarray | None = None,
) -> np.ndarray:
    """source-target 전이율을 질량 보존 반응 행렬 ``Lambda``로 바꾼다.

    Args:
        transition_rates: ``(..., M, M)`` 배열. ``rates[..., m, n]``은
            위상 ``m -> n`` 전이율 ``lambda_mn``이다. 대각 원소는 무시한다.
        transition_mask: 같은 ``(M, M)`` 모양의 0/1 허용 간선 마스크.

    Returns:
        ``(..., M, M)`` 행렬. 첫 행렬 축은 목적 위상, 둘째 행렬 축은
        출발 위상이다. 따라서 ``reaction = Lambda @ rho``이고 각 열의 합은
        정확히 0이어서 전체 위상 질량이 보존된다.
    """

    rates = np.asarray(transition_rates, dtype=np.float64)
    if rates.ndim < 2 or rates.shape[-1] != rates.shape[-2]:
        raise ValueError("transition_rates must have shape (..., M, M)")
    if not np.isfinite(rates).all() or np.any(rates < 0.0):
        raise ValueError("transition rates must be finite and non-negative")

    phase_count = rates.shape[-1]
    if transition_mask is None:
        mask = np.ones((phase_count, phase_count), dtype=np.float64)
        np.fill_diagonal(mask, 0.0)
    else:
        mask = np.asarray(transition_mask, dtype=np.float64)
        if mask.shape != (phase_count, phase_count):
            raise ValueError("transition_mask must have shape (M, M)")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("transition_mask must contain only 0 and 1")
        mask = mask.copy()
        np.fill_diagonal(mask, 0.0)

    masked_rates = rates * mask
    # off-diagonal Lambda[target, source] = lambda[source, target]
    reaction_matrix = np.swapaxes(masked_rates, -1, -2).copy()
    outgoing = masked_rates.sum(axis=-1)
    diagonal = np.arange(phase_count)
    reaction_matrix[..., diagonal, diagonal] = -outgoing
    return reaction_matrix


def reaction_source(
    phase_density: np.ndarray,
    transition_rates: np.ndarray,
    transition_mask: np.ndarray | None = None,
) -> np.ndarray:
    """논문 식 (17)의 ``R(rho) = Lambda(lambda) rho``를 계산한다."""

    density = np.asarray(phase_density, dtype=np.float64)
    matrix = conservative_reaction_matrix(transition_rates, transition_mask)
    if density.shape[-1] != matrix.shape[-1]:
        raise ValueError("phase_density's last axis must contain M phases")
    return np.einsum("...ij,...j->...i", matrix, density)


def macro_adr_residual(
    density_time_derivative: np.ndarray,
    advection_flux_divergence: np.ndarray,
    diffusion_flux_divergence: np.ndarray,
    reaction: np.ndarray,
) -> np.ndarray:
    """논문 식 (25)/(S172)의 점별 Macro-ADR 잔차를 반환한다.

    잔차 정의는 ``d rho/dt + div(u rho) - div(D grad rho) - R(rho)``다.
    네 입력은 같은 모양이어야 한다. 함수가 미분 방법을 강제하지 않으므로
    정규 grid의 유한차분, 자동미분, 또는 해석 미분을 모두 사용할 수 있다.
    """

    terms = [
        np.asarray(density_time_derivative, dtype=np.float64),
        np.asarray(advection_flux_divergence, dtype=np.float64),
        np.asarray(diffusion_flux_divergence, dtype=np.float64),
        np.asarray(reaction, dtype=np.float64),
    ]
    if any(term.shape != terms[0].shape for term in terms[1:]):
        raise ValueError("all Macro-ADR residual terms must have the same shape")
    return terms[0] + terms[1] - terms[2] - terms[3]


def boltzmann_reference_density(
    effective_potential: np.ndarray,
    diffusion: np.ndarray | float,
    *,
    cell_area: float = 1.0,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """논문 식 (27)의 준정상 Boltzmann 기준 밀도를 안정적으로 계산한다.

    ``effective_potential``의 마지막 축이 공간 grid다. 앞쪽 축은 시간이나
    위상 batch로 자유롭게 사용할 수 있다. 각 앞쪽 항목은 공간 적분이 1이
    되도록 별도로 정규화된다.
    """

    potential = np.asarray(effective_potential, dtype=np.float64)
    diffusion_array = np.asarray(diffusion, dtype=np.float64)
    if cell_area <= 0.0:
        raise ValueError("cell_area must be positive")
    if np.any(diffusion_array <= 0.0) or not np.isfinite(diffusion_array).all():
        raise ValueError("diffusion must be finite and strictly positive")

    while diffusion_array.ndim < potential.ndim:
        diffusion_array = np.expand_dims(diffusion_array, axis=-1)
    logits = -potential / diffusion_array

    if support_mask is None:
        support = np.ones_like(potential, dtype=bool)
    else:
        support = np.broadcast_to(np.asarray(support_mask, dtype=bool), potential.shape)
    if np.any(~support.all(axis=-1) & ~support.any(axis=-1)):
        raise ValueError("each density item must contain at least one supported cell")

    # 지원 영역 밖은 exp 계산 전에 -inf로 만들어 정확히 0이 되게 한다.
    masked_logits = np.where(support, logits, -np.inf)
    max_logit = np.max(masked_logits, axis=-1, keepdims=True)
    unnormalized = np.where(support, np.exp(masked_logits - max_logit), 0.0)
    normalizer = np.sum(unnormalized, axis=-1, keepdims=True) * cell_area
    if np.any(normalizer <= 0.0):
        raise ValueError("reference density has empty numerical support")
    return unnormalized / normalizer


def adr_manifold_divergence(
    empirical_density: np.ndarray,
    reference_density: np.ndarray,
    phase_counts: np.ndarray,
    *,
    cell_area: float = 1.0,
) -> np.ndarray:
    """논문 식 (28)의 위상 질량 가중 ``E_ADR``을 계산한다.

    밀도 배열의 마지막 두 축은 ``(..., M, C)``여야 한다. 반환값은 시간 등
    앞쪽 batch 축만 남긴다. 논문처럼 위상별 L2 norm을 먼저 계산한 뒤 위상
    개체 비율로 가중합한다.
    """

    empirical = np.asarray(empirical_density, dtype=np.float64)
    reference = np.asarray(reference_density, dtype=np.float64)
    counts = np.asarray(phase_counts, dtype=np.float64)
    if empirical.shape != reference.shape or empirical.ndim < 2:
        raise ValueError("density arrays must have identical shape (..., M, C)")
    if counts.shape != empirical.shape[:-1]:
        raise ValueError("phase_counts must have shape (..., M)")
    if np.any(counts < 0.0) or cell_area <= 0.0:
        raise ValueError("phase counts must be non-negative and cell_area positive")

    total_count = counts.sum(axis=-1, keepdims=True)
    weights = np.divide(
        counts,
        total_count,
        out=np.zeros_like(counts),
        where=total_count > 0.0,
    )
    phase_l2 = np.sqrt(np.sum((empirical - reference) ** 2, axis=-1) * cell_area)
    return np.sum(weights * phase_l2, axis=-1)
