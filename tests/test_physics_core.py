"""논문 수식의 독립적인 수치 회귀 테스트.

Webots를 실행하지 않아도 Macro-ADR/Micro-EDM의 핵심 불변조건을 검증한다.
이 테스트가 깨지면 시뮬레이션 결과 그래프보다 먼저 물리 구현을 고쳐야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from learning.physics_loss import (
    detached_target_least_squares,
    macro_adr_loss,
    micro_dynamics_loss,
    project_torch_parameters,
    ridge_least_squares_2row,
    soft_cap_nonnegative_loss,
)
from learning.trajectory_loss import (
    exact_macro_adr_loss,
    exact_micro_edm_loss,
    reconstruct_phase_density_torch,
    spatial_divergence_2d,
)
from physics.adr import (
    adr_manifold_divergence,
    boltzmann_reference_density,
    conservative_reaction_matrix,
    macro_adr_residual,
    reaction_source,
)
from physics.density import reconstruct_phase_density, regular_grid
from physics.layouts import (
    LEGACY_PARAMETER_LAYOUTS,
    PAPER_PARAMETER_LAYOUTS,
    get_parameter_layout,
)
from physics.micro_edm import (
    differential_drive_wheel_speeds,
    micro_edm_velocity,
)
from physics.parameters import ParameterLayout, project_numpy_parameters
from tasks.specs import (
    PAPER_SCENARIOS,
    advance_foraging_paper_phases,
    get_runtime_macro_spec,
    get_scenario,
)


def test_scenario_transition_masks_match_declared_edges() -> None:
    for scenario in PAPER_SCENARIOS.values():
        mask = np.asarray(scenario.transition_mask)
        assert mask.shape == (len(scenario.phases), len(scenario.phases))
        assert not np.diag(mask).any()
        assert int(mask.sum()) == len(
            scenario.learned_transitions + scenario.event_transitions
        )

    assert get_scenario("Swarm_Rescue").key == "rescue"
    assert PAPER_SCENARIOS["foraging"].learned_transitions == (
        ("approach", "home"),
        ("home", "explore"),
    )
    assert PAPER_SCENARIOS["rescue"].has_legacy_shape_mismatch
    assert not PAPER_SCENARIOS["foraging"].has_legacy_shape_mismatch
    assert not PAPER_SCENARIOS["navigation"].has_legacy_shape_mismatch
    assert PAPER_SCENARIOS["foraging"].paper_observation_dim == 17
    assert PAPER_SCENARIOS["foraging"].paper_table_observation_dim == 19
    assert PAPER_SCENARIOS["foraging"].paper_enumerated_observation_dim == 18
    assert PAPER_SCENARIOS["navigation"].paper_observation_dim == 16
    assert PAPER_SCENARIOS["navigation"].paper_table_observation_dim == 18
    assert PAPER_SCENARIOS["navigation"].paper_enumerated_observation_dim == 17
    assert PAPER_SCENARIOS["rescue"].paper_observation_dim == 13
    assert PAPER_SCENARIOS["rescue"].paper_table_observation_dim == 14

    foraging_runtime = get_runtime_macro_spec("foraging", phase_contract="paper")
    assert foraging_runtime.all_reaction_edges == (
        (1, 2),
        (2, 0),
        (3, 2),
        (0, 1),
        (0, 3),
    )
    navigation_runtime = get_runtime_macro_spec(
        "navigation", phase_contract="paper"
    )
    assert navigation_runtime.all_reaction_edges == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    )


def test_foraging_paper_phase_machine_only_uses_s129_edges() -> None:
    current = np.array([0, 0, 1, 2, 3, 3, 1])
    next_phase = advance_foraging_paper_phases(
        current,
        carrying=np.array([False, False, True, False, True, False, False]),
        resource_detected=np.array([True, False, False, False, True, True, False]),
        trail_available=np.array([True, True, False, False, False, False, True]),
    )

    # Resource wins over trail in explore; approach/trail only reach home on
    # pickup, and home only returns to explore after drop-off.
    np.testing.assert_array_equal(next_phase, np.array([1, 3, 2, 0, 2, 3, 1]))
    allowed = set(get_runtime_macro_spec("foraging", phase_contract="paper").all_reaction_edges)
    changed = set(zip(current[current != next_phase], next_phase[current != next_phase]))
    assert changed.issubset(allowed)


def test_numpy_parameter_projection_enforces_physical_manifold() -> None:
    layout = ParameterLayout(
        advection_names=("flow", "shape"),
        reaction_names=("enter", "exit"),
        extra_bounds={"beta": (1.0, 5.0)},
        diffusion_bounds=(0.01, 0.04),
        reaction_max=2.0,
        advection_scale=0.13,
    )
    raw = np.array([[100.0, -100.0, -2.0, 0.0, 2.0, 0.0]])
    projected = project_numpy_parameters(raw, layout)

    np.testing.assert_allclose(projected.omega.sum(axis=-1), 1.0)
    assert np.all(projected.omega >= 0.0)
    np.testing.assert_allclose(projected.advection_gains.sum(axis=-1), 0.13)
    assert np.all((projected.diffusion >= 0.01) & (projected.diffusion <= 0.04))
    assert np.all(
        (projected.reaction_rates >= 0.0) & (projected.reaction_rates <= 2.0)
    )
    np.testing.assert_allclose(projected.extras["beta"], [[3.0]])


def test_torch_projection_matches_numpy_and_preserves_gradients() -> None:
    layout = ParameterLayout(
        advection_names=("a", "b", "c"),
        reaction_names=("switch",),
        diffusion_bounds=(0.001, 0.05),
        advection_temperature=2.0,
    )
    raw_torch = torch.tensor(
        [[0.2, -0.1, 0.4, -0.3, 0.7]], dtype=torch.float64, requires_grad=True
    )
    torch_result = project_torch_parameters(raw_torch, layout)
    numpy_result = project_numpy_parameters(raw_torch.detach().numpy(), layout)

    np.testing.assert_allclose(
        torch_result.omega.detach().numpy(), numpy_result.omega, rtol=1e-6
    )
    np.testing.assert_allclose(
        torch_result.diffusion.detach().numpy(), numpy_result.diffusion, rtol=1e-6
    )
    loss = (
        torch_result.advection_gains.square().sum()
        + torch_result.diffusion.sum()
        + torch_result.reaction_rates.sum()
    )
    loss.backward()
    assert raw_torch.grad is not None
    assert torch.isfinite(raw_torch.grad).all()
    assert torch.count_nonzero(raw_torch.grad) > 0


@pytest.mark.parametrize("feature_count", (4, 5))
def test_two_row_ridge_matches_regularized_normal_equations(
    feature_count: int,
) -> None:
    generator = torch.Generator().manual_seed(20260910 + feature_count)
    design = torch.randn(
        19,
        2,
        feature_count,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.randn(
        19, 2, 1, generator=generator, dtype=torch.float64, requires_grad=True
    )
    regularization = 1.0e-4

    gram = design.transpose(-2, -1) @ design
    identity = torch.eye(feature_count, dtype=design.dtype).unsqueeze(0)
    expected = torch.linalg.solve(
        gram + regularization * identity,
        design.transpose(-2, -1) @ target,
    )
    actual = ridge_least_squares_2row(
        design, target, regularization=regularization
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-8, atol=1.0e-9)
    actual.square().mean().backward()
    assert design.grad is not None and torch.isfinite(design.grad).all()
    assert target.grad is not None and torch.isfinite(target.grad).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS device is unavailable"
)
def test_two_row_ridge_runs_and_backpropagates_on_mps() -> None:
    design = torch.randn(4096, 2, 5, device="mps", requires_grad=True)
    target = torch.randn(4096, 2, 1, device="mps", requires_grad=True)

    solution = ridge_least_squares_2row(design, target)
    solution.square().mean().backward()
    torch.mps.synchronize()

    assert torch.isfinite(solution).all().item()
    assert design.grad is not None and torch.isfinite(design.grad).all().item()
    assert target.grad is not None and torch.isfinite(target.grad).all().item()


def test_detached_target_least_squares_matches_torch_cpu() -> None:
    generator = torch.Generator().manual_seed(23)
    design = torch.randn(2, 3, 16, 3, generator=generator, dtype=torch.float64)
    target = torch.randn(2, 3, 16, 1, generator=generator, dtype=torch.float64)
    expected = torch.linalg.lstsq(design, target).solution
    actual = detached_target_least_squares(design, target)
    torch.testing.assert_close(actual, expected, rtol=1.0e-8, atol=1.0e-9)
    assert not actual.requires_grad


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS device is unavailable"
)
def test_detached_target_least_squares_runs_on_mps() -> None:
    design = torch.randn(2, 3, 16, 3, device="mps")
    target = torch.randn(2, 3, 16, 1, device="mps")
    solution = detached_target_least_squares(design, target)
    torch.mps.synchronize()
    assert solution.device.type == "mps"
    assert torch.isfinite(solution).all().item()


def test_named_paper_and_legacy_projection_contracts_are_explicit() -> None:
    """논문 sigmoid 계약과 기존 checkpoint 계약이 차원·수치 모두 추적 가능하다."""

    for scenario_name, scenario in PAPER_SCENARIOS.items():
        assert PAPER_PARAMETER_LAYOUTS[scenario_name].action_dim == scenario.paper_action_dim
        assert LEGACY_PARAMETER_LAYOUTS[scenario_name].action_dim == scenario.legacy_action_dim

    legacy = get_parameter_layout("Swarm_Foraging", contract="legacy")
    raw = np.array(
        [[2.0, 0.3, -0.2, -2.0, 2.0, -2.0, 0.25]], dtype=np.float32
    )
    numpy_result = project_numpy_parameters(raw, legacy)
    torch_result = project_torch_parameters(torch.from_numpy(raw), legacy)

    np.testing.assert_allclose(
        torch_result.omega.numpy(), numpy_result.omega, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        torch_result.diffusion.numpy(), numpy_result.diffusion, rtol=1e-6
    )
    np.testing.assert_allclose(numpy_result.diffusion, [[0.085]])
    np.testing.assert_allclose(numpy_result.reaction_rates, [[0.0, 0.625]])
    np.testing.assert_allclose(numpy_result.omega.sum(axis=-1), 1.0)
    assert np.count_nonzero(numpy_result.omega) < 4

    # 같은 raw D=2라도 paper sigmoid는 경계값에 붙지 않는다.
    paper = get_parameter_layout("foraging", contract="paper")
    paper_result = project_numpy_parameters(raw, paper)
    assert 0.0 < paper_result.diffusion.item() < paper.diffusion_bounds[1]
    assert paper.bounded_transform == "sigmoid"
    assert legacy.bounded_transform == "clipped_linear"


def test_conservative_reaction_matrix_preserves_total_mass() -> None:
    # 0->1, 1->2, 2->0으로 순환하는 서로 다른 전이율을 사용한다.
    rates = np.zeros((3, 3))
    rates[0, 1] = 0.2
    rates[1, 2] = 0.4
    rates[2, 0] = 0.1
    matrix = conservative_reaction_matrix(rates)
    density = np.array([0.5, 0.3, 0.2])
    source = reaction_source(density, rates)

    np.testing.assert_allclose(matrix.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(source.sum(), 0.0, atol=1e-12)
    np.testing.assert_allclose(source, matrix @ density)


def test_phase_kde_integrates_to_phase_population_fraction() -> None:
    positions = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, 0.5]])
    phases = np.array([0, 0, 1])
    points, _, cell_area = regular_grid(
        (-2.0, 2.0), (-2.0, 2.0), resolution=161
    )
    estimate = reconstruct_phase_density(
        positions,
        phases,
        points,
        num_phases=2,
        bandwidth=0.12,
    )
    phase_mass = estimate.density.sum(axis=-1) * cell_area

    np.testing.assert_allclose(phase_mass, [2.0 / 3.0, 1.0 / 3.0], atol=2e-3)
    assert estimate.gradient.shape == (2, len(points), 2)
    assert estimate.laplacian.shape == (2, len(points))


def test_torch_trajectory_kde_matches_numpy_reference() -> None:
    positions = np.array([[-0.3, 0.0], [0.2, 0.1], [0.1, -0.25]])
    phases = np.array([0, 1, 0])
    points, grid_shape, _ = regular_grid(
        (-0.8, 0.8), (-0.7, 0.7), resolution=(11, 9)
    )
    expected = reconstruct_phase_density(
        positions,
        phases,
        points,
        num_phases=2,
        bandwidth=0.18,
    )
    actual = reconstruct_phase_density_torch(
        torch.tensor(positions, dtype=torch.float64)[None, None],
        torch.tensor(phases)[None, None],
        torch.tensor(points, dtype=torch.float64),
        grid_shape,
        num_phases=2,
        bandwidth=0.18,
    )

    np.testing.assert_allclose(
        actual.density[0, 0].reshape(2, -1).numpy(), expected.density, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.gradient[0, 0].reshape(2, -1, 2).numpy(),
        expected.gradient,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.laplacian[0, 0].reshape(2, -1).numpy(),
        expected.laplacian,
        atol=1e-11,
    )


def test_spatial_divergence_is_exact_for_linear_vector_field() -> None:
    points, grid_shape, _ = regular_grid(
        (-1.0, 1.0), (-2.0, 2.0), resolution=(7, 9)
    )
    height, width = grid_shape
    grid = torch.tensor(points, dtype=torch.float64).reshape(height, width, 2)
    field = torch.stack((grid[..., 0], grid[..., 1]), dim=-1)
    divergence = spatial_divergence_2d(
        field,
        x_spacing=2.0 / (width - 1),
        y_spacing=4.0 / (height - 1),
    )

    torch.testing.assert_close(divergence, torch.full_like(divergence, 2.0))


def test_micro_edm_flux_matching_and_drive_saturation() -> None:
    bases = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    weights = np.array([[0.08, 0.02]])
    density = np.array([0.2])
    gradient = np.array([[0.5, -0.25]])
    velocity = micro_edm_velocity(
        bases,
        weights,
        diffusion=np.array([0.03]),
        density=density,
        density_gradient=gradient,
        epsilon=0.05,
    )
    expected = np.array([[0.08, 0.02]]) - 0.03 / 0.25 * gradient
    np.testing.assert_allclose(velocity.desired, expected)

    wheels = differential_drive_wheel_speeds(np.array([[10.0, 10.0]]))
    assert wheels.shape == (1, 2)
    assert np.max(np.abs(wheels)) <= 6.28


def test_exact_micro_loss_compares_executed_velocity_and_backpropagates() -> None:
    bases = torch.tensor(
        [[[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]]]
    )
    gains = torch.tensor([[[[0.08, 0.02], [0.04, 0.06]]]], requires_grad=True)
    diffusion = torch.full((1, 1, 2, 1), 0.01, requires_grad=True)
    density = torch.full((1, 1, 2), 0.2)
    gradient = torch.tensor([[[[0.1, -0.2], [-0.1, 0.05]]]])
    executed = torch.zeros((1, 1, 2, 2))

    result = exact_micro_edm_loss(
        executed,
        bases,
        gains,
        diffusion,
        density,
        gradient,
    )
    assert result.loss.item() > 0.0
    result.loss.backward()
    assert gains.grad is not None and torch.isfinite(gains.grad).all()
    assert diffusion.grad is not None and torch.isfinite(diffusion.grad).all()


def test_exact_macro_loss_reconstructs_kde_and_conserves_reaction_mass() -> None:
    points, grid_shape, _ = regular_grid(
        (-1.0, 1.0), (-1.0, 1.0), resolution=9
    )
    positions = torch.tensor(
        [[
            [[-0.25, 0.0], [0.25, 0.0]],
            [[-0.25, 0.0], [0.25, 0.0]],
            [[-0.25, 0.0], [0.25, 0.0]],
        ]],
        dtype=torch.float64,
    )
    phases = torch.tensor([[[0, 1], [0, 1], [0, 1]]])
    gains = torch.zeros((1, 2, 2, 1), dtype=torch.float64, requires_grad=True)
    diffusion = torch.zeros((1, 2, 2, 1), dtype=torch.float64, requires_grad=True)
    rates = torch.full(
        (1, 2, 2, 2), 0.2, dtype=torch.float64, requires_grad=True
    )
    field_bases = torch.zeros(
        (1, 2, len(points), 1, 2), dtype=torch.float64
    )

    result = exact_macro_adr_loss(
        positions,
        phases,
        torch.tensor(points, dtype=torch.float64),
        grid_shape,
        field_bases,
        gains,
        diffusion,
        rates,
        ((0, 1), (1, 0)),
        num_phases=2,
        bandwidth=0.2,
        dt=0.1,
    )

    torch.testing.assert_close(
        result.density_time_derivative,
        torch.zeros_like(result.density_time_derivative),
    )
    torch.testing.assert_close(
        result.reaction_source.sum(dim=2),
        torch.zeros_like(result.reaction_source[:, :, 0]),
        atol=1e-12,
        rtol=0.0,
    )
    assert result.loss.item() > 0.0
    result.loss.backward()
    assert rates.grad is not None and torch.isfinite(rates.grad).all()

    # Navigation처럼 actor reaction channel이 없는 경우에도 stationary field는
    # 정확히 0 residual이어야 한다.
    no_reaction = exact_macro_adr_loss(
        positions,
        phases,
        torch.tensor(points, dtype=torch.float64),
        grid_shape,
        field_bases,
        gains.detach(),
        diffusion.detach(),
        torch.empty((1, 2, 2, 0), dtype=torch.float64),
        (),
        num_phases=2,
        bandwidth=0.2,
        dt=0.1,
    )
    torch.testing.assert_close(no_reaction.loss, torch.tensor(0.0, dtype=torch.float64))


def test_macro_residual_and_torch_losses_are_zero_for_exact_solution() -> None:
    density_dt = np.array([[0.1, -0.1]])
    advection_div = np.array([[0.2, 0.3]])
    diffusion_div = np.array([[0.05, 0.10]])
    reaction = density_dt + advection_div - diffusion_div
    residual = macro_adr_residual(
        density_dt, advection_div, diffusion_div, reaction
    )
    np.testing.assert_allclose(residual, 0.0)

    zero = torch.zeros((2, 3), requires_grad=True)
    macro_loss = macro_adr_loss(zero, zero, zero, zero)
    micro_loss = micro_dynamics_loss(
        torch.ones((2, 3, 2)), torch.ones((2, 3, 2))
    )
    assert macro_loss.item() == 0.0
    assert micro_loss.item() == 0.0


def test_soft_physics_cap_stays_bounded_without_zeroing_gradient() -> None:
    raw_loss = torch.tensor(1000.0, requires_grad=True)
    bounded = soft_cap_nonnegative_loss(raw_loss, maximum=10.0)
    assert 0.0 < bounded.item() < 10.0
    bounded.backward()
    assert raw_loss.grad is not None and raw_loss.grad.item() > 0.0


def test_adr_reference_normalization_and_divergence() -> None:
    potential = np.array(
        [
            [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    reference = boltzmann_reference_density(
        potential, diffusion=np.array([[0.5, 0.5], [1.0, 1.0]]), cell_area=0.25
    )
    np.testing.assert_allclose(reference.sum(axis=-1) * 0.25, 1.0)

    counts = np.array([[2.0, 1.0], [1.0, 3.0]])
    divergence = adr_manifold_divergence(
        reference, reference.copy(), counts, cell_area=0.25
    )
    np.testing.assert_allclose(divergence, 0.0)
